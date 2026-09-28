import streamlit as st
import time
import plotly.graph_objects as go

from detector import DeepfakeDetector


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Deepfake Video Detector",
    page_icon="🕵",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top,
                #111b4a 0%,
                #080d25 45%,
                #050817 100%
            );
        color: white;
    }

    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: 800;
        color: white;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #aab3d6;
        font-size: 17px;
        margin-bottom: 30px;
    }

    /* RESULT */

    .fake-card {
        background: #35121c;
        border: 2px solid #ff3b6b;
        border-radius: 18px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 0 25px rgba(255, 59, 107, 0.20);
    }

    .real-card {
        background: #102d24;
        border: 2px solid #22d3a6;
        border-radius: 18px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 0 25px rgba(34, 211, 166, 0.20);
    }

    .result-title {
        font-size: 38px;
        font-weight: 800;
        color: white;
        margin-bottom: 12px;
    }

    .result-description {
        font-size: 17px;
        color: #e8ecf8;
        margin-bottom: 12px;
    }

    .result-probability {
        font-size: 27px;
        font-weight: 700;
        color: white;
    }

    /* METRICS */

    .metric-card {
        background: #111a43;
        border: 1px solid #26366e;
        border-radius: 14px;
        padding: 20px 10px;
        text-align: center;
        min-height: 105px;
    }

    .metric-label {
        color: #8e9ac2;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: white;
        font-size: 25px;
        font-weight: 800;
        margin-top: 8px;
    }

    /* SECTION */

    .section-title {
        color: #58f2d2;
        text-align: center;
        font-size: 25px;
        font-weight: 800;
        margin-top: 35px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #7f8bad;
        text-align: center;
        font-size: 13px;
        margin-bottom: 20px;
    }

    .frame-label {
        text-align: center;
        color: #8f9abd;
        font-size: 11px;
        margin-top: 5px;
        margin-bottom: 15px;
    }

    /* INFO */

    .info-box {
        background: #0e1536;
        border: 1px solid #26366e;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
    }

    .info-title {
        color: #58f2d2;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .info-text {
        color: #c4cbe0;
        font-size: 14px;
        line-height: 1.8;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🕵 Deepfake Video Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered video authenticity analysis using VideoMAE</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_detector():
    return DeepfakeDetector()


detector = load_detector()


# =========================================================
# UPLOAD VIDEO
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a video for analysis",
    type=["mp4", "avi", "mov", "mkv", "webm"]
)


if uploaded_file is not None:

    video_bytes = uploaded_file.getvalue()

    temp_video = "temp_uploaded_video.mp4"

    with open(temp_video, "wb") as f:
        f.write(video_bytes)

    st.video(video_bytes)

    st.markdown("")

    analyze_button = st.button(
        "🔍 Analyze Video",
        use_container_width=True,
        type="primary"
    )

    if analyze_button:

        # =================================================
        # PROGRESS
        # =================================================

        progress = st.progress(0)
        status = st.empty()

        status.info("🔄 Extracting video frames...")
        progress.progress(20)

        time.sleep(0.2)

        status.info("🔄 Analyzing temporal video patterns...")
        progress.progress(50)

        try:

            result = detector.analyze(temp_video)

            progress.progress(80)

            status.info("🔄 Calculating confidence metrics...")

            time.sleep(0.2)

            progress.progress(100)

            time.sleep(0.3)

            progress.empty()
            status.empty()

        except Exception as e:

            progress.empty()
            status.empty()

            st.error(
                f"Analysis failed: {str(e)}"
            )

            st.stop()


        # =================================================
        # GET RESULTS
        # =================================================

        prediction = result["prediction"]

        fake_probability = result["fake_probability"]
        real_probability = result["real_probability"]

        confidence = result["confidence"]


        # =================================================
        # RESULT CARD
        # =================================================

        if prediction == "DEEPFAKE":

            st.markdown(
                f'<div class="fake-card">'
                f'<div class="result-title">🚨 DEEPFAKE VIDEO</div>'
                f'<div class="result-description">'
                f'The model detected patterns associated with manipulated video content.'
                f'</div>'
                f'<div class="result-probability">'
                f'Deepfake Probability: {fake_probability * 100:.2f}%'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f'<div class="real-card">'
                f'<div class="result-title">✅ REAL VIDEO</div>'
                f'<div class="result-description">'
                f'The model detected patterns consistent with authentic video content.'
                f'</div>'
                f'<div class="result-probability">'
                f'Real Probability: {real_probability * 100:.2f}%'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )


        # =================================================
        # TOP METRICS
        # =================================================

        col1, col2, col3 = st.columns(3)


        with col1:

            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">FRAMES ANALYZED</div>'
                f'<div class="metric-value">{result["frames_analyzed"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )


        with col2:

            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">PROCESSING TIME</div>'
                f'<div class="metric-value">{result["processing_time"]:.2f}s</div>'
                f'</div>',
                unsafe_allow_html=True
            )


        with col3:

            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">MODEL CONFIDENCE</div>'
                f'<div class="metric-value">{confidence * 100:.2f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )


        # =================================================
        # ANALYZED FRAMES
        # =================================================

        st.markdown(
            '<div class="section-title">Analyzed Frames</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Representative frames extracted and analyzed from the video'
            '</div>',
            unsafe_allow_html=True
        )


        frames = result["display_frames"]

        frame_columns = st.columns(4)


        for i, frame in enumerate(frames):

            with frame_columns[i % 4]:

                # IMPORTANT:
                # No use_container_width here.
                # This works with older Streamlit versions.
                st.image(frame)

                st.markdown(
                    f'<div class="frame-label">Frame {i + 1}</div>',
                    unsafe_allow_html=True
                )


        # =================================================
        # CHARTS
        # =================================================

        chart_col1, chart_col2 = st.columns(2)


        # =================================================
        # PIE CHART
        # =================================================

        with chart_col1:

            st.markdown(
                '<div class="section-title">Confidence Metrics</div>',
                unsafe_allow_html=True
            )

            pie_fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["Deepfake", "Real"],
                        values=[
                            fake_probability * 100,
                            real_probability * 100
                        ],
                        hole=0.55,
                        textinfo="label+percent",
                        textfont=dict(
                            size=15,
                            color="white"
                        ),
                        marker=dict(
                            colors=[
                                "#ff3b6b",
                                "#22d3a6"
                            ],
                            line=dict(
                                color="#080d25",
                                width=3
                            )
                        )
                    )
                ]
            )

            pie_fig.update_layout(
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                showlegend=True,
                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=30
                )
            )

            st.plotly_chart(
                pie_fig,
                use_container_width=True
            )


        # =================================================
        # BAR GRAPH
        # =================================================

        with chart_col2:

            st.markdown(
                '<div class="section-title">Prediction Probability</div>',
                unsafe_allow_html=True
            )

            bar_fig = go.Figure()

            bar_fig.add_trace(
                go.Bar(
                    x=[
                        "Real",
                        "Deepfake"
                    ],
                    y=[
                        real_probability * 100,
                        fake_probability * 100
                    ],
                    text=[
                        f"{real_probability * 100:.2f}%",
                        f"{fake_probability * 100:.2f}%"
                    ],
                    textposition="outside",
                    textfont=dict(
                        color="white",
                        size=16
                    ),
                    marker=dict(
                        color=[
                            "#22d3a6",
                            "#ff3b6b"
                        ]
                    )
                )
            )

            bar_fig.update_layout(
                height=400,
                yaxis=dict(
                    range=[0, 100],
                    title="Probability (%)",
                    gridcolor="rgba(255,255,255,0.08)",
                    color="#aab3d6"
                ),
                xaxis=dict(
                    color="#aab3d6"
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(
                    l=50,
                    r=20,
                    t=30,
                    b=50
                )
            )

            st.plotly_chart(
                bar_fig,
                use_container_width=True
            )


        # =================================================
        # VIDEO INFORMATION
        # =================================================

        st.markdown(
            '<div class="section-title">Video Information</div>',
            unsafe_allow_html=True
        )

        info1, info2, info3, info4 = st.columns(4)


        with info1:
            st.metric(
                "Resolution",
                f'{result["width"]} × {result["height"]}'
            )


        with info2:
            st.metric(
                "FPS",
                f'{result["fps"]:.2f}'
            )


        with info3:
            st.metric(
                "Duration",
                f'{result["duration"]:.2f}s'
            )


        with info4:
            st.metric(
                "Total Frames",
                result["total_frames"]
            )


        # =================================================
        # MODEL INFORMATION
        # =================================================

        st.markdown(
            '<div class="info-box">'
            '<div class="info-title">Model Information</div>'
            '<div class="info-text">'
            '<b>Architecture:</b> VideoMAE<br>'
            '<b>Task:</b> Binary video classification<br>'
            '<b>Classes:</b> Real / Deepfake<br>'
            '<b>Temporal Input:</b> 16 frames per video clip<br>'
            '<b>Input Resolution:</b> 224 × 224<br>'
            '<b>Training Dataset:</b> FaceForensics++ C23<br>'
            '<b>Analysis:</b> Spatial + temporal video patterns'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


        # =================================================
        # INTERPRETATION
        # =================================================

        if prediction == "DEEPFAKE":

            interpretation = (
                "The model assigns a higher probability to the "
                "deepfake class based on the sampled video frames "
                "and temporal visual patterns."
            )

        else:

            interpretation = (
                "The model assigns a higher probability to the "
                "real class based on the sampled video frames "
                "and temporal visual patterns."
            )


        st.markdown(
            f'<div class="info-box">'
            f'<div class="info-title">Analysis Interpretation</div>'
            f'<div class="info-text">'
            f'{interpretation}'
            f'<br><br>'
            f'<b>Important:</b> Model confidence is the predicted '
            f'class probability, not a guarantee that the video '
            f'is authentic or manipulated.'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )