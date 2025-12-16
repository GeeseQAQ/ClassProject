# ==============================================================================
# 导入所需的 Python 库
# ==============================================================================
import streamlit as st          # 导入 Streamlit，这是构建 Web 应用的核心库
import pandas as pd             # 导入 Pandas，用于数据读取、清洗和处理
import plotly.express as px     # 导入 Plotly Express，用于绘制简单、快捷的交互式图表
import plotly.graph_objects as go # 引入 graph_objects 用于画雷达图
from sklearn.cluster import KMeans              # 从 Scikit-learn 导入 K-Means 聚类算法
from sklearn.preprocessing import StandardScaler # 导入标准化工具，用于机器学习前的数据预处理

# ==============================================================================
# 1. 页面基础配置
# ==============================================================================
# 设置页面的标题、图标（浏览器标签页显示）和布局模式
st.set_page_config(
    page_title="中国城市空气质量智能分析系统 (AI版)", # 页面标题
    page_icon="🤖",                                  # 页面图标
    layout="wide"                                    # 布局模式：宽屏显示
)

# ==============================================================================
# 2. 数据加载函数 (带缓存机制)
# ==============================================================================
# 使用装饰器缓存数据，避免每次用户交互（如点击按钮）时都重新读取 CSV，提高运行速度
@st.cache_data
def load_data():
    try:
        # 读取清洗后的 CSV 数据文件
        df = pd.read_csv('china_cities_20251206_cleaned.csv')
        
        # 构造一个标准的时间对象列 (datetime_obj)
        # 逻辑：将 date(20251206) 和 hour(0-23) 拼接成字符串，再转为 datetime 格式
        # .str.zfill(2) 的作用是把 '1' 变成 '01'，确保格式统一
        df['datetime_obj'] = pd.to_datetime(
            df['date'].astype(str) + df['hour'].astype(str).str.zfill(2), 
            format='%Y%m%d%H'
        )
        return df # 返回处理好的 DataFrame
    except Exception as e:
        # 如果读取出错（如文件不存在），在界面显示红色错误信息
        st.error(f"数据加载失败: {e}")
        return pd.DataFrame() # 返回空表防止程序崩溃

# 调用函数加载数据
df = load_data()

# 如果数据为空（加载失败），停止后续代码执行
if df.empty:
    st.stop()

# ==============================================================================
# 3. 数据预处理 (为绘图做准备)
# ==============================================================================
# 定义元数据列名（不需要参与绘图的列）
metadata_cols = ['date', 'hour', 'type', 'datetime_obj']

# 获取所有城市的列名（排除元数据列，并排除可能存在的错误列名）
city_cols = [c for c in df.columns if c not in metadata_cols and c != 'datetime'] 

# 【转换 1：长表格式】 (Long Format)
# 适用于：折线图、柱状图。将"城市"从列名变成一列数据
df_long = df.melt(
    id_vars=metadata_cols,  # 保持不变的列（时间、类型）
    value_vars=city_cols,   # 需要“融化”的列（所有城市）
    var_name='City',        # 新的列名：城市名
    value_name='Value'      # 新的列名：数值
)

# 【转换 2：透视表格式】 (Pivot Table)
# 适用于：相关性分析、散点图、机器学习。每一行是一个(时间,城市)对，列是各种污染物
df_pivot = df_long.pivot_table(
    index=['datetime_obj', 'City'], # 索引
    columns='type',                 # 列：变成 AQI, PM2.5, PM10 等
    values='Value'                  # 值
).reset_index()                     # 重置索引，变回普通 DataFrame

# ==============================================================================
# 4. 侧边栏：AI 智能顾问模块
# ==============================================================================
st.sidebar.title("🤖 AI 智能顾问") # 侧边栏大标题

# 让用户选择所在的城市
user_city = st.sidebar.selectbox("📍 请选择您所在的城市:", city_cols, index=0)

# 获取该用户所选城市的最新一条数据（按时间排序取最后一行）
latest_df = df_pivot[df_pivot['City'] == user_city].sort_values('datetime_obj').iloc[-1]

# 提取关键指标，如果取不到则默认为 0
cur_aqi = latest_df.get('AQI', 0)
cur_pm10 = latest_df.get('PM10', 0)
cur_pm25 = latest_df.get('PM2.5', 0)

# 在侧边栏显示当前 AQI 数值
st.sidebar.markdown(f"**当前 AQI指数**: `{int(cur_aqi)}`")

# --- 规则引擎：根据 AQI 生成建议 ---
adv_color = "green"  # 默认颜色：绿色
adv_text = "空气很好，适合户外活动！🏃" # 默认文案

# 根据 AQI 范围修改文案和颜色
if cur_aqi > 50: adv_text = "空气尚可，敏感人群注意。"; adv_color="orange"
if cur_aqi > 100: adv_text = "轻度污染，建议佩戴口罩。😷"; adv_color="orange"
if cur_aqi > 150: adv_text = "中度污染，减少户外停留。🏠"; adv_color="red"
if cur_aqi > 200: adv_text = "重度污染，严禁户外运动！🚫"; adv_color="red"
if cur_aqi > 300: adv_text = "严重污染，开启空气净化器！🌪️"; adv_color="red"

# --- 特殊规则：沙尘天气判断 ---
# 逻辑：如果 PM10 大于 150 且 PM10 是 PM2.5 的两倍以上，认为是沙尘
if cur_pm10 > 150 and (cur_pm10 / (cur_pm25 + 1) > 2):
    adv_text += "\n\n(检测到沙尘天气特征，请注意防风防沙)"

# 根据颜色显示不同级别的提示框
if adv_color == "green": st.sidebar.success(adv_text)
elif adv_color == "orange": st.sidebar.warning(adv_text)
else: st.sidebar.error(adv_text)

# 分割线
st.sidebar.markdown("---")

# ==============================================================================
# 5. 侧边栏：全局数据筛选器
# ==============================================================================
st.sidebar.header("🎛️ 数据筛选")
# 多选框：选择要对比的城市，默认选中北上广西
selected_cities = st.sidebar.multiselect("对比分析城市:", city_cols, default=["北京", "上海", "西安", "广州"])
# 下拉框：选择要分析的主要指标（如 AQI, PM2.5）
pollutant_type = st.sidebar.selectbox("主要分析指标:", df['type'].unique(), index=0)

# ==============================================================================
# 6. 主界面：标题与 Tabs 布局
# ==============================================================================
st.title("中国城市空气质量数据挖掘与智能分析") # 主标题
st.markdown("本系统集成 **机器学习聚类** 与 **多维统计分析**，从 6 个维度深度解读空气质量数据。") # 副标题/说明

# 创建 6 个标签页，分别对应不同的分析维度
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📍 空间排名", "📈 时间趋势", "🔗 关联分析", "🔬 成分结构", "📊 质量分布", "🧪 AI聚类挖掘"
])

# ==============================================================================
# Tab 1: 空间维度 (柱状图排名)
# ==============================================================================
with tab1:
    st.subheader(f"🏙️ 空间维度：{pollutant_type} 城市排名")
    
    # 删除了分栏 (st.columns)，直接展示
    df_rank = df_long[df_long['type'] == pollutant_type].groupby('City')['Value'].mean().sort_values(ascending=False)
    rank_mode = st.radio("查看模式", ["Top 15 污染", "Top 15 清洁"], horizontal=True)
    
    plot_data = df_rank.head(15) if rank_mode == "Top 15 污染" else df_rank.tail(15).sort_values()
    
    fig1 = px.bar(
        x=plot_data.index, 
        y=plot_data.values, 
        color=plot_data.values, 
        color_continuous_scale='RdYlGn_r' if rank_mode=="Top 15 清洁" else 'Reds',
        labels={'x': '城市名称', 'y': f'{pollutant_type} 平均数值'},
        text_auto='.1f'
    )
    fig1.update_layout(xaxis_tickangle=0)
    st.plotly_chart(fig1, use_container_width=True)

# ==============================================================================
# Tab 2: 时间维度 (折线图趋势)
# ==============================================================================
with tab2:
    st.subheader(f"🕰️ 时间维度：{pollutant_type} 24H 变化")
    # 检查是否选择了城市，如果没选则提示警告
    if selected_cities:
        # 筛选数据：只保留选中的指标和选中的城市
        df_trend = df_long[(df_long['type'] == pollutant_type) & (df_long['City'].isin(selected_cities))].sort_values('datetime_obj')
        
        # 绘制折线图
        fig2 = px.line(
            df_trend, 
            x='datetime_obj', 
            y='Value', 
            color='City',   # 不同城市不同颜色
            markers=True,   # 显示数据点标记
            # 设置中文标签映射
            labels={
                'datetime_obj': '监测时间 (2025-12-06)', 
                'Value': f'{pollutant_type} 监测数值',
                'City': '城市名称'
            }
        )
        
        # 强制 X 轴水平显示
        fig2.update_layout(xaxis_tickangle=0) 
        st.plotly_chart(fig2, use_container_width=True)
    else: 
        st.warning("请在侧边栏选择城市")

# ==============================================================================
# Tab 3: 关联维度 (热力图)
# ==============================================================================
with tab3:
    st.subheader("🔗 关联维度：污染物相关性矩阵")
    # 定义所有可能的污染物列名
    valid_cols = [p for p in ['AQI', 'PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3'] if p in df_pivot.columns]
    
    # 如果数据中有超过1种污染物，才能画相关性图
    if len(valid_cols) > 1:
        # 计算相关系数矩阵 (.corr())，并绘制热力图 (imshow)
        fig3 = px.imshow(df_pivot[valid_cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig3, use_container_width=True)

# ==============================================================================
# Tab 4: 结构维度 (散点图 PM2.5 vs PM10)
# ==============================================================================
with tab4:
    st.subheader("🔬 结构维度：PM2.5/PM10 成分分析")
    # 检查是否有这两列数据
    if 'PM2.5' in df_pivot.columns and 'PM10' in df_pivot.columns:
        # 绘制散点图
        fig4 = px.scatter(
            df_pivot, 
            x='PM10', 
            y='PM2.5', 
            color='AQI',         # 点的颜色代表 AQI 高低
            hover_name='City',   # 鼠标悬停显示城市名
            title="颗粒物结构分布", 
            opacity=0.6,
            labels={'PM10': 'PM10 浓度 (μg/m³)', 'PM2.5': 'PM2.5 浓度 (μg/m³)'} # 中文轴标签
        )
        # 添加一条对角虚线 (x=y)，用于辅助判断
        fig4.add_shape(type="line", x0=0, y0=0, x1=500, y1=500, line=dict(color="Gray", dash="dash"))
        st.plotly_chart(fig4, use_container_width=True)

# ==============================================================================
# Tab 5: 分布维度 (饼图)
# ==============================================================================
with tab5:
    st.subheader("📊 分布维度：空气质量等级占比")
    
    # 辅助函数：根据 AQI 数值返回等级名称
    def get_level(aqi):
        if aqi <= 50: return '优'
        elif aqi <= 100: return '良'
        elif aqi <= 150: return '轻度'
        elif aqi <= 200: return '中度'
        elif aqi <= 300: return '重度'
        else: return '严重'

    if 'AQI' in df_pivot.columns:
        # 计算每个等级出现的次数
        counts = df_pivot['AQI'].dropna().apply(get_level).value_counts().reset_index()
        counts.columns = ['Level', 'Count'] # 重命名列
        
        # 绘制饼图
        st.plotly_chart(px.pie(counts, values='Count', names='Level', color_discrete_sequence=px.colors.sequential.RdBu_r), use_container_width=True)

# ==============================================================================
# Tab 6: 机器学习聚类 K-Means
# ==============================================================================
with tab6:
    st.subheader("🧪 ：基于K-Means的城市污染模式自动识别")
    st.markdown("""
    > **双视图解析**：
    > * **🕸️ 雷达图**：看“体型”，快速识别是偏科（沙尘）还是全面发展（综合污染）。
    > * **📊 柱状图**：看“身高”，了解具体的污染物浓度数值。
    """)

    col_ml1, col_ml2 = st.columns([1, 3])
    
    # --- 左侧：参数控制 ---
    with col_ml1:
        # 范围扩大到 2-8，默认设为 5
        n_clusters = st.slider("聚类数量 (K值)", 2, 8, 6) 
        st.info("💡 **建议**：\n设置 K=5 或 6 可以最好地分离出“交通型”和“浮尘型”城市。")
    
    # --- 1. 数据准备 (严谨清洗) ---
    ml_features = ['AQI', 'PM2.5', 'PM10', 'CO', 'NO2', 'SO2']
    ml_features = [f for f in ml_features if f in df_pivot.columns]
    
    # 严谨模式：直接剔除缺失值
    df_city_features = df_pivot.groupby('City')[ml_features].mean().dropna()

    if df_city_features.empty:
        st.error("有效数据不足，无法进行聚类。")
    else:
        # --- 2. 训练模型 ---
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_city_features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_city_features['Cluster_ID'] = kmeans.fit_predict(data_scaled)
        
        # --- 3. 智能打标逻辑 (Pro版：规则更具体) ---
        cluster_means = df_city_features.groupby('Cluster_ID')[ml_features].mean()
        
        def get_cluster_name(row):
            aqi = row.get('AQI', 0)
            pm10 = row.get('PM10', 0)
            pm25 = row.get('PM2.5', 0)
            so2 = row.get('SO2', 0)
            no2 = row.get('NO2', 0)
            co = row.get('CO', 0)
            
            # 计算关键比值
            ratio_pm = pm10 / (pm25 + 0.1) 

            # --- 规则引擎 (优先级从高到低) ---
            if aqi < 35: return "🍃 极优生态型"
            if pm10 > 200 and ratio_pm > 2.5: return "🏜️ 强沙尘暴型"
            if pm10 > 120 and ratio_pm > 2.0: return "🌪️ 浮尘扬沙型"
            if so2 > 25: return "🏭 重工业燃煤型"
            if so2 > 15 and co > 1.0: return "🏗️ 轻工业/取暖型"
            if no2 > 45: return "🚗 交通拥堵型"
            if aqi > 200: return "🔴 极重复合污染"
            if aqi > 150: return "🟣 重度雾霾型"
            if aqi > 100: return "🟠 轻度雾霾型"
            if aqi < 70: return "🌿 清洁宜居型"
            return "🔵 综合过渡型"

        label_map = {}
        for idx, row in cluster_means.iterrows():
            label_map[idx] = get_cluster_name(row)
            
        df_city_features['Label'] = df_city_features['Cluster_ID'].map(label_map)

        # --- 4. 可视化：3D 总览图 ---
        with col_ml2:
            st.caption(f"有效分析城市: **{len(df_city_features)}** 个 | 模式细分度: **{n_clusters} 级**")
            fig_3d = px.scatter_3d(
                df_city_features.reset_index(),
                x='PM10', y='PM2.5', z='AQI',
                color='Label',
                hover_name='City',
                title="城市污染模式 3D 精细聚类",
                color_discrete_sequence=px.colors.qualitative.Bold,
                labels={'Label': '细分模式'}
            )
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=450)
            st.plotly_chart(fig_3d, use_container_width=True)

        # --- 5. 核心展示：雷达图 + 柱状图 (双图并列) ---
        st.markdown("### 🧬 污染基因图谱 (双视图)")
        
        # 归一化 (0-1) 用于雷达图
        df_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
        df_norm = df_norm.fillna(0)
        
        # 动态布局：如果K>4，分成两行显示
        cols = st.columns(n_clusters) if n_clusters <= 4 else st.columns(4)
        
        for i, (cluster_id, label) in enumerate(label_map.items()):
            # 处理多行布局 (换行逻辑)
            col_idx = i % 4
            if i >= 4 and col_idx == 0: 
                cols = st.columns(n_clusters - 4)
            
            with cols[col_idx]:
                st.markdown(f"**{label}**")
                cities = df_city_features[df_city_features['Cluster_ID'] == cluster_id].index.tolist()
                
                # 智能显示代表城市
                priority = ['北京', '上海', '西安', '喀什地区', '三亚', '唐山', '武汉']
                shown_cities = [c for c in cities if c in priority] + [c for c in cities if c not in priority]
                st.caption(f"包含: {', '.join(shown_cities[:3])} 等")
                
                # --- 图表 A: 雷达图 (看形态) ---
                r_vals = df_norm.loc[cluster_id].tolist(); r_vals += [r_vals[0]]
                theta_vals = df_norm.columns.tolist(); theta_vals += [theta_vals[0]]
                
                # 智能配色
                color_code = '#636EFA' 
                if "沙尘" in label: color_code = '#FFA15A'
                if "工业" in label: color_code = '#EF553B'
                if "清洁" in label or "极优" in label: color_code = '#00CC96'
                if "交通" in label: color_code = '#AB63FA'

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_vals, theta=theta_vals, fill='toself',
                    line_color=color_code, opacity=0.6
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=140,
                    showlegend=False,
                    title=dict(text="特征形态", font=dict(size=12), y=0.95)
                )
                st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
                
                # --- 图表 B: 柱状图 (看数值 - 已加回) ---
                real_vals = cluster_means.loc[cluster_id]
                fig_bar = px.bar(
                    x=real_vals.index, y=real_vals.values,
                    color=real_vals.index,
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_bar.update_layout(
                    showlegend=False, xaxis_tickangle=0,
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=140, # 高度适中
                    title=dict(text="均值浓度", font=dict(size=12), y=0.95),
                    xaxis_title=None, yaxis_title=None
                )
                # 隐藏Y轴刻度，保留网格线，界面更清爽
                fig_bar.update_yaxes(showticklabels=False, showgrid=True)
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})