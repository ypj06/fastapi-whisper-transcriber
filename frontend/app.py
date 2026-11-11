"""Streamlit 前端。

启动：
    streamlit run frontend/app.py

默认连接 http://localhost:8000 的后端，可在侧边栏修改。
"""

from __future__ import annotations

import base64
import os

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
LANGUAGES = ["en", "zh", "ja", "ko", "fr", "de"]
PRESET_LABELS = {
    "mobile": "📱 Mobile — 手机端，低延迟",
    "desktop_high_quality": "🖥️ Desktop HQ — 桌面离线，最高质量",
    "noisy_environment": "🔊 Noisy — 工厂/户外，激进降噪",
    "meeting_room": "🏢 Meeting — 会议室，去混响",
    "lightweight": "⚡ Lightweight — 极速，嵌入式",
}

st.set_page_config(page_title="Whisper 转写 & 音频预处理", layout="wide")


# --------------------------------------------------------------------- 工具函数
def api_health(api_url: str) -> dict | None:
    """探活：拿不到 /health 就认为后端未启动。"""
    try:
        resp = requests.get(f"{api_url}/health", timeout=3)
        return resp.json() if resp.ok else None
    except requests.RequestException:
        return None


def fmt_preset(name: str) -> str:
    return PRESET_LABELS.get(name, name)


# --------------------------------------------------------------------- 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    api_url = st.text_input("后端地址", value=DEFAULT_API_URL)

    health = api_health(api_url)
    if health is None:
        st.error("后端未连接")
    else:
        st.success(f"后端正常（{health['device']}）")
        if not health["model_loaded"]:
            st.warning("模型尚未加载")
        if not health["translation_enabled"]:
            st.info("未配置百度翻译密钥，翻译将返回原文")

    model_opt = st.selectbox(
        "选择模型",
        options=[1, 2, 3, 4, 5, 6, 7],
        index=1,
        format_func=lambda x: f"{x} → {['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'][x - 1]}",
    )
    languages = st.multiselect("识别语言", LANGUAGES, default=["en"])
    target_lang = st.selectbox(
        "字幕翻译目标语言（可选）",
        options=[None, *LANGUAGES],
        format_func=lambda x: "不翻译" if x is None else x,
    )
    output_formats = st.multiselect("输出格式", ["SRT 字幕", "纯文本 TXT"], default=["SRT 字幕", "纯文本 TXT"])
    overwrite = st.checkbox("强制覆写已有文件", value=False)

    if st.button("加载模型", use_container_width=True):
        with st.spinner("加载中…"):
            try:
                resp = requests.post(f"{api_url}/model/load", params={"model_choice": model_opt}, timeout=600)
            except requests.RequestException as exc:
                st.error(f"请求失败：{exc}")
            else:
                if resp.ok:
                    st.success(resp.json()["message"])
                else:
                    st.error(f"模型加载失败：{resp.text}")

# --------------------------------------------------------------------- 主体
st.title("🎙️ Whisper 音视频转写 & 音频预处理")
st.caption("Faster-Whisper 转写 · 百度翻译双语字幕 · 降噪/VAD/增强预处理")
st.divider()

tab_batch, tab_single, tab_preprocess, tab_compare = st.tabs(
    ["📁 批量目录转写", "📤 单个文件转写", "🧹 音频预处理", "🔀 预处理+转写对比"]
)

# ---- Tab 1 批量 -------------------------------------------------------------
with tab_batch:
    st.subheader("批量处理目录下的所有音视频")
    dir_path = st.text_input("音视频目录路径（服务端路径）", value="./samples")
    if st.button("开始批量转写", type="primary"):
        payload = {
            "directory": dir_path,
            "languages": languages or ["en"],
            "target_lang": target_lang,
            "model_choice": model_opt,
            "gen_srt": "SRT 字幕" in output_formats,
            "gen_txt": "纯文本 TXT" in output_formats,
            "overwrite": overwrite,
        }
        with st.spinner("处理中…"):
            try:
                resp = requests.post(f"{api_url}/transcribe/batch", json=payload, timeout=3600)
            except requests.RequestException as exc:
                st.error(f"请求失败：{exc}")
            else:
                if resp.ok:
                    data = resp.json()
                    st.text("\n".join(data["log"]))
                    st.success(
                        f"完成：成功 {data['succeeded']} / 跳过 {data['skipped']} / 失败 {data['failed']}"
                    )
                else:
                    st.error(f"错误：{resp.text}")

# ---- Tab 2 单文件 -----------------------------------------------------------
with tab_single:
    st.subheader("上传单个音视频文件")
    uploaded_file = st.file_uploader(
        "选择文件", type=["mp4", "mkv", "mp3", "wav", "m4a", "flac"], key="transcribe_upload"
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        lang_single = st.selectbox("识别语言", ["auto", *LANGUAGES], index=1)
    with col_b:
        target_lang_single = st.selectbox(
            "翻译目标语言", options=[None, *LANGUAGES], format_func=lambda x: "不翻译" if x is None else x
        )
    with col_c:
        out_format = st.radio("输出格式", ["srt", "txt", "both"], index=2, horizontal=True)

    bilingual = st.checkbox("翻译时输出双语字幕（原文 + 译文）", value=False)

    if uploaded_file and st.button("开始转写", type="primary"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        params = {
            "language": lang_single,
            "target_lang": target_lang_single,
            "output_format": out_format,
            "bilingual": bilingual,
        }
        with st.spinner("转写中…"):
            try:
                resp = requests.post(f"{api_url}/transcribe/file", files=files, params=params, timeout=3600)
            except requests.RequestException as exc:
                st.error(f"请求失败：{exc}")
            else:
                if resp.ok:
                    data = resp.json()
                    cols = st.columns(2)
                    with cols[0]:
                        if data.get("srt"):
                            st.download_button(
                                "⬇️ 下载 SRT", data["srt"], file_name=f"{uploaded_file.name}.srt"
                            )
                    with cols[1]:
                        if data.get("txt"):
                            st.download_button(
                                "⬇️ 下载 TXT", data["txt"], file_name=f"{uploaded_file.name}.txt"
                            )
                    st.text_area("转写文本", value=data.get("txt", ""), height=240)
                    with st.expander(f"带时间戳的分段（{len(data.get('segments', []))} 条）"):
                        for seg in data.get("segments", []):
                            st.markdown(f"`{seg['start']:.1f}s – {seg['end']:.1f}s` {seg['text']}")
                else:
                    st.error(f"错误：{resp.text}")

# ---- Tab 3 预处理 -----------------------------------------------------------
with tab_preprocess:
    st.subheader("🧹 音频预处理（降噪 + VAD + 语音增强）")
    st.markdown("上传带噪声的录音 → 选择场景预设 → 得到干净音频，可直接试听与下载。")

    preset_info = {}
    try:
        resp = requests.get(f"{api_url}/preprocess/presets", timeout=5)
        preset_info = resp.json() if resp.ok else {}
    except requests.RequestException:
        pass
    preset_options = list(preset_info) or list(PRESET_LABELS)

    left, right = st.columns([1, 1])
    with left:
        uploaded_noisy = st.file_uploader(
            "上传音频或视频",
            type=["wav", "mp3", "m4a", "flac", "ogg", "mp4", "mkv", "webm", "flv", "avi", "mov"],
            key="preprocess_upload",
        )
        preset_choice = st.selectbox("预处理预设", preset_options, format_func=fmt_preset, key="preset_select")
        with st.expander("🔧 高级调整"):
            nr_strength = st.slider("降噪强度", 0.0, 1.0, 0.85, 0.05)
            enable_vad = st.checkbox("启用 VAD", value=True)
            enable_nr = st.checkbox("启用降噪", value=True)
            enable_se = st.checkbox("启用信号增强", value=True)
        process_clicked = st.button("🚀 运行预处理", type="primary", use_container_width=True)

    with right:
        st.markdown("##### 预设速查")
        st.table(
            {
                "场景": ["会议室多人", "嘈杂环境", "手机/边缘", "离线最高质量", "最小延迟"],
                "预设": ["meeting_room", "noisy_environment", "mobile", "desktop_high_quality", "lightweight"],
            }
        )
        if preset_choice in preset_info:
            with st.expander("📋 当前预设配置"):
                st.json(preset_info[preset_choice])

    if uploaded_noisy and process_clicked:
        files = {"file": (uploaded_noisy.name, uploaded_noisy.getvalue())}
        params = {"preset": preset_choice, "nr_strength": nr_strength}
        if not enable_vad:
            params["enable_vad"] = "false"
        if not enable_nr:
            params["enable_nr"] = "false"
        if not enable_se:
            params["enable_se"] = "false"

        with st.spinner("正在预处理…"):
            try:
                resp = requests.post(f"{api_url}/preprocess/audio", files=files, params=params, timeout=1800)
            except requests.RequestException as exc:
                st.error(f"请求失败：{exc}")
            else:
                if not resp.ok:
                    st.error(f"预处理失败：{resp.text}")
                else:
                    data = resp.json()
                    clean_audio = base64.b64decode(data["processed_audio_base64"])
                    stats = data["stats"]

                    st.success(f"完成，耗时 {stats['processing_time_total_ms']:.0f} ms")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("原始时长", f"{data['original_duration_s']:.1f}s")
                    c2.metric("处理后时长", f"{data['processed_duration_s']:.1f}s")
                    c3.metric("VAD 片段", stats["vad_segments"])
                    c4.metric("输出 RMS", f"{stats['output_rms_db']:.1f} dB")

                    audio_left, audio_right = st.columns(2)
                    with audio_left:
                        st.markdown("**原始（含噪声）**")
                        st.audio(uploaded_noisy.getvalue())
                    with audio_right:
                        st.markdown("**处理后**")
                        st.audio(clean_audio, format="audio/wav")

                    st.download_button(
                        "⬇️ 下载干净音频 (WAV)",
                        clean_audio,
                        file_name=f"clean_{data['filename']}.wav",
                        mime="audio/wav",
                    )
                    with st.expander("📊 完整统计"):
                        st.json(stats)

# ---- Tab 4 对比 -------------------------------------------------------------
with tab_compare:
    st.subheader("🔀 预处理 + 转写对比")
    st.markdown("同一条音频分别用原始波形和预处理后波形转写，直观对比降噪对识别的影响。")

    uploaded_compare = st.file_uploader(
        "上传音频文件", type=["mp4", "mkv", "mp3", "wav", "m4a"], key="compare_upload"
    )

    col_left, col_right = st.columns([1, 1])
    with col_left:
        compare_preset = st.selectbox(
            "预处理预设", preset_options, format_func=fmt_preset, key="compare_preset"
        )
        compare_language = st.selectbox("识别语言", LANGUAGES, index=0, key="compare_lang")
    with col_right:
        with st.expander("🔧 高级设置"):
            c_nr = st.slider("降噪强度", 0.0, 1.0, 0.85, 0.05, key="c_nr")
            c_vad = st.checkbox("VAD", value=True, key="c_vad")
            c_nr_enable = st.checkbox("降噪", value=True, key="c_nr_enable")
            c_se = st.checkbox("信号增强", value=True, key="c_se")

    if uploaded_compare and st.button("🔄 预处理 + 转写", type="primary", use_container_width=True):
        files = {"file": (uploaded_compare.name, uploaded_compare.getvalue())}
        params = {
            "preset": compare_preset,
            "language": compare_language,
            "output_format": "both",
            "nr_strength": c_nr,
        }
        if not c_vad:
            params["enable_vad"] = "false"
        if not c_nr_enable:
            params["enable_nr"] = "false"
        if not c_se:
            params["enable_se"] = "false"

        with st.spinner("预处理并转写中…"):
            try:
                resp = requests.post(
                    f"{api_url}/preprocess-and-transcribe", files=files, params=params, timeout=3600
                )
            except requests.RequestException as exc:
                st.error(f"请求失败：{exc}")
            else:
                if resp.ok:
                    st.session_state["compare_data"] = resp.json()
                else:
                    st.error(f"请求失败：{resp.text}")

    data = st.session_state.get("compare_data")
    if data:
        stats = data["stats"]
        c1, c2, c3 = st.columns(3)
        c1.metric("原始时长", f"{stats['input_duration_s']:.1f}s")
        c2.metric("处理后时长", f"{stats['output_duration_s']:.1f}s")
        c3.metric("处理耗时", f"{stats['processing_time_total_ms']:.0f}ms")

        st.divider()
        st.markdown("### 📝 转写对比")
        base_col, enh_col = st.columns(2)
        with base_col:
            st.markdown("#### 🔴 原始音频")
            st.text_area("原始转写", value=data["comparison"]["baseline_text"], height=200, disabled=True)
            st.download_button(
                "⬇️ 下载原始 TXT",
                data["comparison"]["baseline_text"],
                file_name="baseline.txt",
            )
        with enh_col:
            st.markdown("#### 🟢 预处理后")
            st.text_area("预处理后转写", value=data["comparison"]["enhanced_text"], height=200, disabled=True)
            st.download_button(
                "⬇️ 下载预处理后 TXT",
                data["comparison"]["enhanced_text"],
                file_name="enhanced.txt",
            )

        base_srt = (data["baseline"] or {}).get("srt") or ""
        enh_srt = (data["enhanced"] or {}).get("srt") or ""
        if base_srt or enh_srt:
            with st.expander("SRT 字幕"):
                srt_col1, srt_col2 = st.columns(2)
                srt_col1.text(base_srt)
                srt_col2.text(enh_srt)

        with st.expander("📋 详细时间戳分段"):
            seg_left, seg_right = st.columns(2)
            with seg_left:
                st.markdown("**原始时间轴**")
                for seg in (data["baseline"] or {}).get("segments", []):
                    st.markdown(f"`{seg['start']:.1f}s – {seg['end']:.1f}s` {seg['text']}")
            with seg_right:
                st.markdown("**预处理后时间轴**")
                for seg in (data["enhanced"] or {}).get("segments", []):
                    st.markdown(f"`{seg['start']:.1f}s – {seg['end']:.1f}s` {seg['text']}")
