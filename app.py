import io
from datetime import datetime
from xml.sax.saxutils import escape
from pathlib import Path
import joblib
import librosa
import numpy as np
import pandas as pd
import streamlit as st
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import queue
import av
import threading
from streamlit_autorefresh import st_autorefresh


st.set_page_config(page_title='SOUNDTRACE', page_icon='🎧', layout='wide')
MODEL_DIR = Path(__file__).parent / 'models'
MODEL_NAMES = {
    'FAN1': 'FAN1_LOF17_Final.joblib',
    'FAN2': 'FAN2_LOF22A_Final.joblib',
    'FAN3': 'FAN3_LOF17_Final.joblib',
    'REF1': 'REF1_LOF17_Final.joblib',
    'REF2': 'REF2_LOF15B_Final.joblib',
    'REF3': 'REF3_LOF17_Final.joblib',
}
RF_FEATURES = ['RMS', 'Spectral_Centroid'] + [f'MFCC_{i}' for i in range(1,14)]

class LiveAudioBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.samples = np.array([], dtype=np.float32)
        self.sample_rate = None
        self.chunks = queue.Queue(maxsize=30)

    def add_frame(self, frame):
        audio = frame.to_ndarray()

        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        audio = audio.astype(np.float32)

        if frame.format.name.startswith("s16"):
            audio /= 32768.0
        elif frame.format.name.startswith("s32"):
            audio /= 2147483648.0

        sr = frame.sample_rate

        with self.lock:
            if self.sample_rate != sr:
                self.samples = np.array([], dtype=np.float32)
                self.sample_rate = sr

            self.samples = np.concatenate((self.samples, audio))

            segment_size = int(sr * 2)

            while len(self.samples) >= segment_size:
                segment = self.samples[:segment_size].copy()
                self.samples = self.samples[segment_size:]

                try:
                    self.chunks.put_nowait((segment, sr))
                except queue.Full:
                    try:
                        self.chunks.get_nowait()
                        self.chunks.put_nowait((segment, sr))
                    except queue.Empty:
                        pass


def band_energy(segment, sr, low, high):
    spectrum = np.fft.rfft(segment.astype(np.float64))
    freq = np.fft.rfftfreq(len(segment), d=1/sr)
    power = np.abs(spectrum)**2
    mask = (freq >= low) & (freq <= high)
    return float(power[mask].sum()/(power[1:].sum()+1e-10))


def extract_features(segment, sr):
    rms = float(librosa.feature.rms(y=segment).mean())
    centroid = float(librosa.feature.spectral_centroid(y=segment,sr=sr).mean())
    mfcc = librosa.feature.mfcc(y=segment,sr=sr,n_mfcc=13).mean(axis=1)
    peak = float(np.max(np.abs(segment)))
    exact_rms = float(np.sqrt(np.mean(segment.astype(np.float64)**2)))
    result = {
        'RMS': rms, 'Spectral_Centroid': centroid,
        'Peak_Amplitude': peak,
        'Crest_Factor': peak/(exact_rms+1e-10),
        'Energy_180Hz': band_energy(segment,sr,170,190),
        'Energy_Low_1': band_energy(segment,sr,60,150),
        'Energy_Low_2': band_energy(segment,sr,150,300),
        'Energy_Mid_1': band_energy(segment,sr,300,600),
        'Energy_Mid_2': band_energy(segment,sr,600,1000),
    }
    for i,v in enumerate(mfcc,1):
        result[f'MFCC_{i}'] = float(v)
    return result


def segment_features(y, sr, feature_names):
    n = int(2*sr)
    count = len(y)//n
    if count == 0:
        raise ValueError('최소 2초 이상의 오디오가 필요합니다.')
    rows = [extract_features(y[i*n:(i+1)*n],sr) for i in range(count)]
    frame = pd.DataFrame(rows)
    missing = set(feature_names) - set(frame.columns)
    if missing:
        raise ValueError(f'모델 특징이 누락되었습니다: {sorted(missing)}')
    return frame.loc[:,feature_names]


@st.cache_resource(show_spinner='AI 모델 불러오는 중...')
def load_models():
    rf_package = joblib.load(MODEL_DIR/'RandomForest_6Device_Final.joblib')
    if isinstance(rf_package,dict):
        rf = rf_package['model']
        rf_features = rf_package.get('features', RF_FEATURES)
    else:
        rf = rf_package
        rf_features = RF_FEATURES
    if not hasattr(rf, 'predict_proba'):
        raise ValueError('RandomForest 모델에 predict_proba가 없습니다.')
    lof = {name:joblib.load(MODEL_DIR/filename) for name,filename in MODEL_NAMES.items()}
    return rf, list(rf_features), lof


def load_audio(data):
    # librosa.load keeps native sample rate; mono downmix is explicit.
    return librosa.load(io.BytesIO(data), sr=None, mono=True)


def predict(data):
    rf, rf_features, lof_packages = load_models()
    y, sr = load_audio(data)
    if not np.isfinite(y).all() or not np.isfinite(sr) or sr <= 0:
        raise ValueError('유효한 오디오 파일이 아닙니다.')
    duration = len(y)/sr
    if duration < 2:
        raise ValueError('2초 이상 녹음해 주세요. 정확한 분석을 위해 20~25초를 권장합니다.')
    y_rf = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
    rf_data = segment_features(y_rf,16000,rf_features)
    proba = rf.predict_proba(rf_data.to_numpy()).mean(axis=0)
    idx = int(np.argmax(proba))
    device = str(rf.classes_[idx])
    if device not in lof_packages:
        raise ValueError(f'기기 식별 결과를 지원하지 않습니다: {device}')
    package = lof_packages[device]
    lof_features = list(package['features'])
    lof_data = segment_features(y,sr,lof_features)
    scores = -package['model'].score_samples(package['scaler'].transform(lof_data.to_numpy()))
    threshold = float(package['threshold'])
    abnormal = scores > threshold
    consecutive = int(package.get('consecutive_alert_segments',3))
    run = 0
    alert = False
    for flag in abnormal:
        run = run+1 if flag else 0
        if run >= consecutive:
            alert = True
    df = pd.DataFrame({
        '시간(초)': np.arange(len(scores))*2,
        '이상 점수': scores,
        '판정 임계값': threshold,
        '이상 구간': abnormal,
    })
    return {
        'device':device, 'confidence':float(proba[idx]),
        'probabilities':pd.DataFrame({'기기':rf.classes_.astype(str),'식별 확률':proba}).sort_values('식별 확률',ascending=False),
        'model':package.get('model_name',MODEL_NAMES[device]),
        'duration':duration,'sr':sr,'alert':alert,
        'abnormal_count':int(abnormal.sum()),'total_count':len(abnormal),
        'threshold':threshold,'segments':df,
    }


def predict_live_segment(y, sr):
    rf, rf_features, lof_packages = load_models()

    y = np.asarray(y, dtype=np.float32)

    if sr <= 0 or not np.isfinite(y).all():
        raise ValueError('유효하지 않은 오디오 데이터입니다.')

    if len(y) < sr * 2:
        raise ValueError('2초 이상의 오디오가 필요합니다.')

    # RandomForest로 가전제품 식별
    y_rf = librosa.resample(
        y, orig_sr=sr, target_sr=16000
    ) if sr != 16000 else y

    rf_data = segment_features(
        y_rf, 16000, rf_features
    )

    proba = rf.predict_proba(
        rf_data.to_numpy()
    ).mean(axis=0)

    idx = int(np.argmax(proba))
    device = str(rf.classes_[idx])
    confidence = float(proba[idx])

    # 가전제품 식별 신뢰도 확인 (시험 기준 80%)
    if confidence < 0.80:
        return {
            'device': '식별 불확실',
            'confidence': confidence,
            'score': None,
            'threshold': None,
            'abnormal': False,
            'uncertain': True
        }


    # 해당 가전제품의 LOF 모델 선택
    package = lof_packages[device]
    lof_features = list(package['features'])

    lof_data = segment_features(
        y, sr, lof_features
    )

    scores = -package['model'].score_samples(
        package['scaler'].transform(
            lof_data.to_numpy()
        )
    )

    threshold = float(package['threshold'])
    score = float(scores[0])
    abnormal = bool(score > threshold)

    return {
        'device': device,
        'confidence': confidence,
        'score': score,
        'threshold': threshold,
        'abnormal': abnormal,
        'uncertain': False
    }


# PDF report generator (ReportLab). Keeps CSV exports unchanged.
def _pdf_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    candidates = [
        Path('C:/Windows/Fonts/malgun.ttf'),
        Path('C:/Windows/Fonts/NanumGothic.ttf'),
        Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'),
        Path('/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'),
    ]
    for font_path in candidates:
        if font_path.exists():
            if 'SoundtraceKorean' not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont('SoundtraceKorean', str(font_path)))
            return 'SoundtraceKorean'
    raise RuntimeError('PDF 한글 글꼴을 찾을 수 없습니다. Windows의 맑은 고딕 글꼴을 확인해 주세요.')


def make_pdf_report(rows, mode_name, metadata=None):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.graphics.shapes import Drawing, Line, String, Rect

    font = _pdf_font()
    metadata = metadata or {}
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=42, leftMargin=42,
                            topMargin=45, bottomMargin=48)
    navy = colors.HexColor('#183153')
    blue = colors.HexColor('#2474A6')
    light = colors.HexColor('#F0F5FA')
    muted = colors.HexColor('#64748B')
    styles = {
        'title': ParagraphStyle('STTitle', fontName=font, fontSize=20, leading=27, textColor=navy, spaceAfter=8),
        'sub': ParagraphStyle('STSub', fontName=font, fontSize=12, leading=18, textColor=navy, spaceBefore=13, spaceAfter=8),
        'body': ParagraphStyle('STBody', fontName=font, fontSize=9, leading=15, textColor=navy),
        'small': ParagraphStyle('STSmall', fontName=font, fontSize=7.4, leading=12, textColor=navy),
        'head': ParagraphStyle('STHead', fontName=font, fontSize=8, leading=12, textColor=colors.white),
    }
    def para(value, style='body'):
        return Paragraph(escape(str(value)), styles[style])

    total = len(rows)
    uncertain = sum(bool(r.get('uncertain', False)) for r in rows)
    abnormal = sum(bool(r.get('abnormal', False)) and not bool(r.get('uncertain', False)) for r in rows)
    normal = total - uncertain - abnormal
    story = [para('SOUNDTRACE', 'title'), para('AI 기반 생활가전 이상 소음 분석 보고서'), Spacer(1, 8)]
    details = [('분석 방식', mode_name), ('보고서 생성 시각', datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S'))]
    details.extend((str(k), str(v)) for k, v in metadata.items())
    info = Table([[para(k,'small'), para(v,'small')] for k,v in details], colWidths=[120, 385], hAlign='LEFT')
    info.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),light),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7)]))
    story.extend([info, para('분석 결과 요약','sub')])
    summary = Table([[para('전체 구간','small'),para('정상 패턴','small'),para('이상 패턴','small'),para('식별 불확실','small')],
                     [para(total),para(normal),para(abnormal),para(uncertain)]], colWidths=[126]*4)
    summary.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),light),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#D6E1EA')),('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
    story.append(summary)
    if total:
        story.append(Spacer(1,9))
        story.append(para(f'정상 {normal/total*100:.1f}%  |  이상 {abnormal/total*100:.1f}%  |  식별 불확실 {uncertain/total*100:.1f}%', 'small'))

    # Draw a native vector plot, so no extra plotting dependency is needed.
    values = [(i, float(r['score']), float(r['threshold'])) for i,r in enumerate(rows)
              if r.get('score') is not None and r.get('threshold') is not None
              and np.isfinite(float(r['score'])) and np.isfinite(float(r['threshold']))]
    story.append(para('구간별 이상 점수 추이','sub'))
    if values:
        w,h=500,155
        plot=Drawing(w,h)
        left,right,bottom,top=48,485,28,140
        nums=[v for _,score,thr in values for v in (score,thr)]
        lo,hi=min(nums),max(nums)
        if hi<=lo: hi=lo+1.0
        margin=(hi-lo)*0.10
        lo-=margin; hi+=margin
        plot.add(Line(left,bottom,left,top,strokeColor=muted,strokeWidth=0.6))
        plot.add(Line(left,bottom,right,bottom,strokeColor=muted,strokeWidth=0.6))
        for t in range(4):
            yy=bottom+(top-bottom)*t/3
            val=lo+(hi-lo)*t/3
            plot.add(Line(left,yy,right,yy,strokeColor=colors.HexColor('#E6EDF3'),strokeWidth=0.5))
            plot.add(String(2,yy-2,f'{val:.2f}',fontName=font,fontSize=7,fillColor=muted))
        def point(i,v):
            return (left+(right-left)*i/max(1,total-1),bottom+(top-bottom)*(v-lo)/(hi-lo))
        for key,color in [('score',blue),('threshold',colors.HexColor('#D88736'))]:
            previous=None
            for i,r in enumerate(rows):
                v=r.get(key)
                if v is None or not np.isfinite(float(v)):
                    previous=None
                    continue
                xy=point(i,float(v))
                if previous is not None:
                    plot.add(Line(previous[0],previous[1],xy[0],xy[1],strokeColor=color,strokeWidth=1.5))
                previous=xy
        plot.add(Line(65,10,87,10,strokeColor=blue,strokeWidth=2))
        plot.add(String(91,7,'이상 점수',fontName=font,fontSize=8,fillColor=navy))
        plot.add(Line(185,10,207,10,strokeColor=colors.HexColor('#D88736'),strokeWidth=2))
        plot.add(String(211,7,'판정 임계값',fontName=font,fontSize=8,fillColor=navy))
        story.append(plot)
        story.append(para('가로축: 분석 구간 순서(각 2초). 식별 불확실 구간은 점수가 없어 선이 끊깁니다.', 'small'))
    else:
        story.append(para('이상 점수를 계산할 수 있는 구간이 없어 그래프를 표시하지 않습니다.','small'))

    story.append(para('구간별 상세 기록','sub'))
    headers=['번호','시각 / 위치','식별 기기','확률','이상 점수','임계값','판정']
    def fmt(v):
        return '-' if v is None or pd.isna(v) else f'{float(v):.3f}'
    table_rows=[[para(h,'head') for h in headers]]
    for i,r in enumerate(rows,1):
        verdict='판단 보류' if r.get('uncertain') else ('이상 패턴' if r.get('abnormal') else '정상 패턴')
        table_rows.append([para(i,'small'),para(r.get('time','-'),'small'),para(r.get('device','-'),'small'),
                           para(f"{float(r.get('confidence',0))*100:.1f}%",'small'),para(fmt(r.get('score')),'small'),
                           para(fmt(r.get('threshold')),'small'),para(verdict,'small')])
    t=Table(table_rows,colWidths=[30,107,77,48,72,72,98],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),
                           ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
                           ('LINEBELOW',(0,0),(-1,0),0.6,navy)]))
    story.append(t)
    story.append(para('결과 해석 및 유의사항','sub'))
    if uncertain:
        story.append(para(f'총 {uncertain}개 구간은 식별 확률이 임시 기준(60%) 미만이어서 이상 여부 판단을 보류했습니다. 판단 보류는 정상 판정을 의미하지 않습니다.','small'))
    if abnormal:
        story.append(para('이상 패턴 구간은 학습된 정상 소음과의 차이를 나타냅니다. 실제 기계 고장을 확정하지 않습니다.','small'))
    else:
        story.append(para('이상 패턴이 확인되지 않았더라도 실제 기기의 정상 작동이나 무고장을 보증하지 않습니다.','small'))
    story.append(para('본 보고서는 연구용 PoC 결과입니다. 주변 소음, 마이크, 녹음 환경에 따라 결과가 달라질 수 있으며, 식별 확률은 정확도 보증값이 아닙니다.','small'))
    def footer(canvas, pdfdoc):
        canvas.saveState()
        canvas.setFont(font,8)
        canvas.setFillColor(muted)
        canvas.drawString(42,26,'SOUNDTRACE · 연구용 분석 보고서')
        canvas.drawRightString(A4[0]-42,26,f'{pdfdoc.page} 페이지')
        canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return buf.getvalue()


st.title('🎧 SOUNDTRACE')
st.caption('AI 기반 생활가전 이상 소음 조기 감지 · 시제품(PoC)')
mode = st.radio('분석 모드', ['실시간 감시', '수동 분석'], index=1, horizontal=True)


if mode.startswith('실시간'):
    st.subheader('실시간 자동 감시')
    st.write('마이크 소리를 2초 단위로 자동 분석합니다.')

    if 'live_audio_buffer' not in st.session_state:
        st.session_state['live_audio_buffer'] = LiveAudioBuffer()

    if 'live_results' not in st.session_state:
        st.session_state['live_results'] = []

    if 'live_alert_count' not in st.session_state:
        st.session_state['live_alert_count'] = 0

    audio_buffer = st.session_state['live_audio_buffer']

    def audio_frame_callback(frame):
        audio_buffer.add_frame(frame)
        return frame

    ctx = webrtc_streamer(
        key='soundtrace-microphone',
        mode=WebRtcMode.SENDONLY,
        audio_frame_callback=audio_frame_callback,
        media_stream_constraints={
            'video': False,
            'audio': True
        },
        async_processing=True
    )

    if ctx.state.playing:
        st.success('마이크가 연결되었습니다. 자동 분석 중입니다.')

        # 2초마다 화면 갱신
        st_autorefresh(
            interval=2000,
            key='soundtrace-auto-refresh'
        )

        # 수집된 2초 오디오를 순서대로 분석
        processed = 0

        while processed < 5:
            try:
                chunk = audio_buffer.chunks.get_nowait()
            except queue.Empty:
                break

            try:
                # chunk는 (오디오 배열, 샘플링 주파수) 형식
                y, sr = chunk
                result = predict_live_segment(y, sr)
                result['analyzed_at'] = datetime.now().astimezone().isoformat(timespec='seconds')

                st.session_state['live_results'].append(result)

                # 식별 불확실 구간은 연속 이상 판정을 끊습니다.
                if result.get('uncertain', False):
                    st.session_state['live_alert_count'] = 0
                elif result['abnormal']:
                    st.session_state['live_alert_count'] += 1
                else:
                    st.session_state['live_alert_count'] = 0

            except Exception as e:
                st.error(f'AI 분석 오류: {e}')

            processed += 1

        results = st.session_state['live_results']

        if results:
            latest = results[-1]

            st.divider()
            st.subheader('실시간 AI 분석 결과')
            a, b, c = st.columns(3)
            a.metric('식별된 가전제품', latest['device'])
            b.metric('기기 식별 확률', f"{latest['confidence'] * 100:.1f}%")

            if latest.get('uncertain', False):
                c.metric('이상 점수', '분석 보류')
                st.info('기기 식별이 불확실합니다. 이상 소음 판단을 보류합니다.')
            else:
                c.metric('이상 점수', f"{latest['score']:.4f}")
                st.write(f"판정 임계값: {latest['threshold']:.4f}")

                if st.session_state['live_alert_count'] >= 3:
                    st.error('🚨 3개 연속 구간에서 이상 소음이 감지되었습니다!')
                elif latest['abnormal']:
                    st.warning('⚠️ 이상 소음 패턴 감지 (연속 확인 중)')
                else:
                    st.success('✅ 현재 구간은 정상 패턴입니다.')

            st.write('연속 이상 구간:', st.session_state['live_alert_count'])
            st.write('총 분석 구간 수:', len(results))

            # 최근 50개 구간 점수 그래프 (식별 불확실 구간은 빈 값)
            recent = results[-50:]
            chart_df = pd.DataFrame({
                '이상 점수': [
                    r['score'] if r['score'] is not None else np.nan
                    for r in recent
                ],
                '판정 임계값': [
                    r['threshold'] if r['threshold'] is not None else np.nan
                    for r in recent
                ],
            })

            if chart_df['이상 점수'].notna().any():
                st.line_chart(chart_df)
            else:
                st.info('아직 식별 가능한 가전제품 소리가 없어 이상 점수 그래프를 표시하지 않습니다.')
        else:
            st.info('첫 번째 2초 오디오 분석을 기다리고 있습니다.')

    else:
        st.info('START 버튼을 누르면 자동 감시를 시작합니다.')

        if st.button('이전 분석 결과 초기화'):
            st.session_state['live_results'] = []
            st.session_state['live_alert_count'] = 0
            st.rerun()

    # 마이크를 중지한 후에도 이전 분석 기록을 다운로드할 수 있습니다.
    if st.session_state['live_results']:
        st.divider()
        st.subheader('실시간 분석 기록 CSV 저장')
        export_rows = []
        for number, item in enumerate(st.session_state['live_results'], start=1):
            uncertain = bool(item.get('uncertain', False))
            if uncertain:
                verdict = '식별 불확실 - 판단 보류'
            elif item['abnormal']:
                verdict = '이상 패턴 구간'
            else:
                verdict = '정상 패턴 구간'
            export_rows.append({
                '구간 번호': number,
                '분석 시각': item.get('analyzed_at', ''),
                '식별 기기': item['device'],
                '식별 확률(%)': round(item['confidence'] * 100, 2),
                '이상 점수': item['score'],
                '판정 임계값': item['threshold'],
                '판정 결과': verdict,
                '식별 불확실': uncertain,
            })
        export_df = pd.DataFrame(export_rows)
        csv_bytes = export_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            '📥 실시간 분석 기록 CSV 다운로드',
            data=csv_bytes,
            file_name=f"SOUNDTRACE_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
            use_container_width=True,
            key='download_live_csv'
        )
        st.caption('현재 화면에 보관된 모든 분석 구간을 저장합니다. 초기화하거나 앱을 종료하기 전에 다운로드하세요.')
        pdf_rows = [{
            'time': item.get('analyzed_at', f'{i*2}초'),
            'device': item['device'], 'confidence': item['confidence'],
            'score': item.get('score'), 'threshold': item.get('threshold'),
            'abnormal': item.get('abnormal',False), 'uncertain': item.get('uncertain',False)
        } for i,item in enumerate(st.session_state['live_results'])]
        try:
            pdf_bytes = make_pdf_report(pdf_rows, '실시간 감시', {'분석 간격':'2초'})
            st.download_button('📄 실시간 분석 PDF 보고서 다운로드', data=pdf_bytes,
                file_name=f"SOUNDTRACE_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime='application/pdf', use_container_width=True, key='download_live_pdf')
        except Exception as e:
            st.warning(f'PDF 생성 오류: {e}')


    st.stop()




method = st.radio('분석 방법', ['직접 녹음','파일 업로드'], horizontal=True)
if method == '직접 녹음':
    st.write('가전제품 가까이에서 약 20~25초 녹음해 주세요. 마이크 사용 권한을 허용해야 합니다.')
    audio = st.audio_input('가전제품 소리 녹음', sample_rate=48000)
else:
    audio = st.file_uploader('WAV 파일을 업로드해 주세요',type=['wav'])

if audio is not None:
    data = audio.getvalue()
    st.audio(data,format='audio/wav')
    if st.button('🔍 AI 분석하기',type='primary',use_container_width=True):
        try:
            with st.spinner('기기 식별 및 이상 소음 분석 중...'):
                result = predict(data)
            st.session_state['last_result'] = result
            st.session_state['last_audio_name'] = getattr(audio,'name','직접 녹음')
        except Exception as e:
            st.session_state.pop('last_result',None)
            st.error(f'분석 중 오류가 발생했습니다: {e}')
else:
    st.session_state.pop('last_result',None)

if 'last_result' in st.session_state:
    r = st.session_state['last_result']
    st.divider()
    st.subheader('AI 분석 결과')
    a,b,c = st.columns(3)
    a.metric('식별 기기',r['device'])
    b.metric('기기 식별 확률',f"{r['confidence']*100:.1f}%")
    c.metric('분석 결과','⚠️ 이상 경고' if r['alert'] else '✅ 연속 이상 경고 없음')
    st.caption(f"분석 길이 {r['duration']:.1f}초 · 입력 {r['sr']:,}Hz · 이상 구간 {r['abnormal_count']}/{r['total_count']} · 모델 {r['model']}")
    if r['alert']:
        st.warning('3개 이상의 연속된 2초 구간에서 이상 소음 패턴이 감지되었습니다.')
    else:
        st.success('3개 연속 이상 구간이 확인되지 않았습니다. 일부 구간의 이상 점수는 높을 수 있습니다.')
    st.subheader('시간대별 이상 점수')
    st.line_chart(r['segments'].set_index('시간(초)')[['이상 점수','판정 임계값']])
    with st.expander('구간별 분석 상세'):
        st.dataframe(r['segments'],use_container_width=True)
    with st.expander('기기 식별 확률'):
        st.dataframe(r['probabilities'],use_container_width=True)
    # 수동 분석의 구간별 결과도 별도 CSV로 저장할 수 있습니다.
    manual_df = r['segments'].copy()
    manual_df.insert(0, '식별 기기', r['device'])
    manual_df.insert(1, '기기 식별 확률(%)', round(r['confidence'] * 100, 2))
    manual_df.insert(2, '원본 파일명', st.session_state.get('last_audio_name', ''))
    st.download_button(
        '📥 수동 분석 구간별 CSV 다운로드',
        data=manual_df.to_csv(index=False).encode('utf-8-sig'),
        file_name=f"SOUNDTRACE_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime='text/csv',
        use_container_width=True,
        key='download_manual_csv'
    )
    manual_rows = [{
        'time': f"{float(item['시간(초)']):.0f}초",
        'device': r['device'], 'confidence': r['confidence'],
        'score': float(item['이상 점수']), 'threshold': float(item['판정 임계값']),
        'abnormal': bool(item['이상 구간']), 'uncertain': False
    } for _,item in r['segments'].iterrows()]
    try:
        pdf_bytes = make_pdf_report(manual_rows, '수동 분석', {
            '원본 파일':st.session_state.get('last_audio_name',''),
            '녹음 길이':f"{r['duration']:.1f}초",
            '식별 기기':r['device'],
            '연속 이상 경고':'발생' if r['alert'] else '없음',
            '사용 모델':r['model'],
        })
        st.download_button('📄 수동 분석 PDF 보고서 다운로드',data=pdf_bytes,
            file_name=f"SOUNDTRACE_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime='application/pdf',use_container_width=True,key='download_manual_pdf')
    except Exception as e:
        st.warning(f'PDF 생성 오류: {e}')
    st.caption('본 결과는 정상 소음 패턴과의 차이를 탐지하는 연구용 PoC 결과이며 실제 고장 여부를 확진하지 않습니다. 마이크·주변 소음·샘플링 환경에 따라 결과가 달라질 수 있습니다.')
