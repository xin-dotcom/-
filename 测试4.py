import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import pandas as pd
from pyngrok import ngrok
import nest_asyncio
import torch
# 在代码开头添加ngrok认证（需注册获取免费token）
NGROK_AUTH_TOKEN = "2wz8Z2WGx0vTbQgHr2aOvrA9VX1_4LkABYA4yP5AUojhLdizb"  # 从https://dashboard.ngrok.com获取
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

torch.cuda.empty_cache()  # 释放未使用的显存（网页4）
nest_asyncio.apply()
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_WATCHER"] = "false"

# 页面配置
st.set_page_config(
    page_title="肿瘤检测分析系统",
    page_icon="🩺",
    layout="centered"
)

# 页面标题
st.title("医学影像智能分析平台")
st.markdown("---")
st.subheader("智分瘤影联盟 | 精准医疗解决方案")


# 模型加载优化
@st.cache_resource
def load_seg_model():
    model_path = r'D:\PythonProject1\last.pt'
    if not os.path.exists(model_path):
        st.error("❌ 模型加载失败：请检查模型文件路径")
        st.stop()
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"模型初始化失败：{str(e)}")
        st.stop()


def format_coordinates(box):
    """转换坐标格式为(左, 上, 宽, 高)"""
    x_center, y_center, width, height = box.xywh[0].tolist()
    left = x_center - (width / 2)
    top = y_center - (height / 2)
    return (left, top), (width, height)


def calculate_areas(box, mask):
    """计算边界框和分割区域面积"""
    bbox_area = box.xywh[0][2] * box.xywh[0][3]
    mask_area = np.sum(mask.data.cpu().numpy()) if mask is not None else 0
    return int(bbox_area), int(mask_area)


def main():
    # 文件上传组件
    uploaded_file = st.file_uploader(
        "上传医学影像（支持CT/PET/病理切片）",
        type=["png", "jpg", "jpeg"],
        help="建议上传DICOM格式或高清PNG图像"
    )

    if uploaded_file:
        # 双列布局
        col_img, col_data = st.columns([2, 3])

        with col_img:
            # 原始影像显示
            st.image(uploaded_file,
                     caption="原始医学影像",
                     use_column_width=True,
                     clamp=True)

        # 分析处理流程
        if st.button("🚀 开始智能分析", type="primary"):
            with st.spinner("AI分析中...预计耗时5-10秒"):
                try:
                    # 初始化模型
                    model = load_seg_model()

                    # 创建临时工作区
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        # 预处理图像
                        img = Image.open(uploaded_file).convert("RGB")
                        tmp_path = os.path.join(tmp_dir, "input.png")
                        img.save(tmp_path)

                        # 执行预测
                        results = model.predict(
                            source=tmp_path,
                            save=False,
                            show=False,
                            device='0',
                            retina_masks=True,
                            imgsz=640
                        )
                        result = results[0]

                        # 结果可视化
                        plotted_img = result.plot()
                        plotted_img = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)

                        # 生成分析报告
                        report_data = []
                        for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
                            # 坐标转换
                            (left, top), (width, height) = format_coordinates(box)

                            # 面积计算
                            bbox_area, mask_area = calculate_areas(box, mask)

                            # 数据记录
                            report_data.append({
                                "病变类型": model.names[int(box.cls)],
                                "置信度": float(box.conf),
                                "左上坐标": f"({int(left)}, {int(top)})",
                                "病灶尺寸": f"{int(width)}×{int(height)}",
                                "边界框面积": bbox_area,
                                "病变区域面积": mask_area
                            })

                        # 生成CSX报告
                        df_report = pd.DataFrame(report_data)
                        csv_path = os.path.join(tmp_dir, "analysis_report.csv")
                        df_report.to_csv(csv_path, index=False)

                        with col_data:
                            # 显示检测结果
                            st.image(plotted_img,
                                     caption="AI标注结果",
                                     use_column_width=True,
                                     output_format="PNG")

                            # 显示数据报告
                            st.subheader("📊 定量分析报告")
                            st.dataframe(
                                df_report.style.format({
                                    "置信度": "{:.2%}",  # ✅ 正确应用百分比格式
                                    "边界框面积": "{:,}",
                                    "病变区域面积": "{:,}"
                                }),
                                height=300,
                                use_container_width=True
                            )

                            # 结果下载
                            with open(csv_path, "rb") as f:
                                st.download_button(
                                    label="📥 下载完整报告",
                                    data=f,
                                    file_name=f"肿瘤分析_{uploaded_file.name.split('.')[0]}.csv",
                                    mime="text/csv",
                                    help="包含详细量化指标的CSV格式报告"
                                )

                except Exception as e:
                    st.error(f"分析中断：{str(e)}")
                    st.error("可能原因：1) GPU内存不足 2) 图像格式异常 3) 模型文件损坏")

# 在预测完成后添加
torch.cuda.empty_cache()
# 修改main函数后的执行部分
if __name__ == "__main__":
    # 清理现有隧道
    try:
        tunnels = ngrok.get_tunnels()
        for tunnel in tunnels:
            ngrok.disconnect(tunnel.public_url)
        print(f"已清理 {len(tunnels)} 个旧隧道")
    except Exception as e:
        st.sidebar.warning(f"隧道清理失败: {str(e)}")

    # 创建新隧道
    try:
        public_url = ngrok.connect(addr='8501', proto='http')
        st.sidebar.success(f"🌐 远程访问地址：\n{public_url}")
        st.sidebar.info("首次访问需1-2分钟初始化")
    except Exception as e:
        st.error(f"❌ 隧道创建失败: {str(e)}")
        st.error("""
            常见原因：
            1. 网络连接不稳定
            2. Ngrok账户隧道数超限（免费账户限3个）
            3. 端口8501被占用
            解决方案：
            - 等待5分钟后重试
            - 访问 https://dashboard.ngrok.com/endpoints/status 手动关闭旧隧道
            - 重启路由器更换公网IP
        """)
        st.stop()

    # 启动主程序
    main()