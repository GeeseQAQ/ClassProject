import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# 1. 页面配置与工具函数
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="中国城市空气质量智能分析系统 (AI版)",
    page_icon="🤖",
    layout="wide"
)

# 加载数据
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('china_cities_20251206_cleaned.csv')
        df['datetime_obj'] = pd.to_datetime(
            df['date'].astype(str) + df['hour'].astype(str).str.zfill(2), 
            format='%Y%m%d%H'
        )
        return df
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# 数据预处理
metadata_cols = ['date', 'hour', 'type', 'datetime_obj']
# 排除可能存在的 'datetime' 列
city_cols = [c for c in df.columns if c not in metadata_cols and c != 'datetime'] 
df_long = df.melt(id_vars=metadata_cols, value_vars=city_cols, var_name='City', value_name='Value')
df_pivot = df_long.pivot_table(index=['datetime_obj', 'City'], columns='type', values='Value').reset_index()

# -----------------------------------------------------------------------------
# 2. 侧边栏：智能顾问
# -----------------------------------------------------------------------------
st.sidebar.title("🤖 AI 智能顾问")

user_city = st.sidebar.selectbox("📍 请选择您所在的城市:", city_cols, index=0)

latest_df = df_pivot[df_pivot['City'] == user_city].sort_values('datetime_obj').iloc[-1]
cur_aqi = latest_df.get('AQI', 0)
cur_pm10 = latest_df.get('PM10', 0)
cur_pm25 = latest_df.get('PM2.5', 0)

st.sidebar.markdown(f"**当前 AQI指数**: `{int(cur_aqi)}`")

adv_color = "green"
adv_text = "空气很好，适合户外活动！🏃"
if cur_aqi > 50: adv_text = "空气尚可，敏感人群注意。"; adv_color="orange"
if cur_aqi > 100: adv_text = "轻度污染，建议佩戴口罩。😷"; adv_color="orange"
if cur_aqi > 150: adv_text = "中度污染，减少户外停留。🏠"; adv_color="red"
if cur_aqi > 200: adv_text = "重度污染，严禁户外运动！🚫"; adv_color="red"
if cur_aqi > 300: adv_text = "严重污染，开启空气净化器！🌪️"; adv_color="red"

if cur_pm10 > 150 and (cur_pm10 / (cur_pm25 + 1) > 2):
    adv_text += "\n\n(检测到沙尘天气特征，请注意防风防沙)"

if adv_color == "green": st.sidebar.success(adv_text)
elif adv_color == "orange": st.sidebar.warning(adv_text)
else: st.sidebar.error(adv_text)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ 数据筛选")
selected_cities = st.sidebar.multiselect("对比分析城市:", city_cols, default=["北京", "上海", "西安", "广州"])
pollutant_type = st.sidebar.selectbox("主要分析指标:", df['type'].unique(), index=0)

# -----------------------------------------------------------------------------
# 3. 主界面
# -----------------------------------------------------------------------------
st.title("中国城市空气质量数据挖掘与智能分析")
st.markdown("本系统集成 **机器学习聚类** 与 **多维统计分析**，从 6 个维度深度解读空气质量数据。")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📍 空间排名", "📈 时间趋势", "🔗 关联分析", "🔬 成分结构", "📊 质量分布", "🧪 AI聚类挖掘"
])

# --- Tab 1: 空间排名 (已去除提示信息，图表全宽显示) ---
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

# --- Tab 2: 时间趋势 ---
with tab2:
    st.subheader(f"🕰️ 时间维度：{pollutant_type} 24H 变化")
    if selected_cities:
        df_trend = df_long[(df_long['type'] == pollutant_type) & (df_long['City'].isin(selected_cities))].sort_values('datetime_obj')
        
        fig2 = px.line(
            df_trend, 
            x='datetime_obj', 
            y='Value', 
            color='City', 
            markers=True,
            labels={
                'datetime_obj': '监测时间 (2025-12-06)', 
                'Value': f'{pollutant_type} 监测数值',
                'City': '城市名称'
            }
        )
        
        fig2.update_layout(xaxis_tickangle=0) 
        st.plotly_chart(fig2, use_container_width=True)
    else: st.warning("请在侧边栏选择城市")

# --- Tab 3: 关联分析 ---
with tab3:
    st.subheader("🔗 关联维度：污染物相关性矩阵")
    valid_cols = [p for p in ['AQI', 'PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3'] if p in df_pivot.columns]
    if len(valid_cols) > 1:
        fig3 = px.imshow(df_pivot[valid_cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig3, use_container_width=True)

# --- Tab 4: 成分结构 ---
with tab4:
    st.subheader("🔬 结构维度：PM2.5/PM10 成分分析")
    if 'PM2.5' in df_pivot.columns and 'PM10' in df_pivot.columns:
        fig4 = px.scatter(
            df_pivot, 
            x='PM10', 
            y='PM2.5', 
            color='AQI', 
            hover_name='City', 
            title="颗粒物结构分布", 
            opacity=0.6,
            labels={'PM10': 'PM10 浓度 (μg/m³)', 'PM2.5': 'PM2.5 浓度 (μg/m³)'}
        )
        fig4.add_shape(type="line", x0=0, y0=0, x1=500, y1=500, line=dict(color="Gray", dash="dash"))
        st.plotly_chart(fig4, use_container_width=True)

# --- Tab 5: 质量分布 ---
with tab5:
    st.subheader("📊 分布维度：空气质量等级占比")
    def get_level(aqi):
        if aqi <= 50: return '优'
        elif aqi <= 100: return '良'
        elif aqi <= 150: return '轻度'
        elif aqi <= 200: return '中度'
        elif aqi <= 300: return '重度'
        else: return '严重'
    if 'AQI' in df_pivot.columns:
        counts = df_pivot['AQI'].dropna().apply(get_level).value_counts().reset_index()
        counts.columns = ['Level', 'Count']
        st.plotly_chart(px.pie(counts, values='Count', names='Level', color_discrete_sequence=px.colors.sequential.RdBu_r), use_container_width=True)

# --- Tab 6: 机器学习聚类 ---
with tab6:
    st.subheader("🧪 基于 K-Means 的城市污染模式挖掘")
    col_ml1, col_ml2 = st.columns([1, 3])
    
    with col_ml1:
        n_clusters = st.slider("选择聚类数量 (K值)", 2, 6, 4)
        st.markdown("**聚类依据特征**:")
        st.code("AQI, PM2.5, PM10,\nCO, NO2, SO2", language="text")
        
    ml_features = ['AQI', 'PM2.5', 'PM10', 'CO', 'NO2', 'SO2']
    ml_features = [f for f in ml_features if f in df_pivot.columns]
    
    df_city_features = df_pivot.groupby('City')[ml_features].mean().dropna()
    
    if not df_city_features.empty:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_city_features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_city_features['Cluster'] = kmeans.fit_predict(data_scaled)
        df_city_features['Cluster'] = df_city_features['Cluster'].astype(str)
        
        with col_ml2:
            x_ax = 'PM10'
            y_ax = 'PM2.5'
            z_ax = 'AQI'
            
            if all(col in df_city_features.columns for col in [x_ax, y_ax, z_ax]):
                fig_ml = px.scatter_3d(
                    df_city_features.reset_index(),
                    x=x_ax, y=y_ax, z=z_ax,
                    color='Cluster',
                    hover_name='City',
                    title=f"城市污染模式 3D 聚类图 (K={n_clusters})",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    labels={'Cluster': '聚类类别', 'City': '城市'}
                )
                fig_ml.update_layout(margin=dict(l=0, r=0, b=0, t=30))
                st.plotly_chart(fig_ml, use_container_width=True)

        st.markdown("### 🧬 聚类结果深度解码")
        cluster_analysis = df_city_features.groupby('Cluster')[ml_features].mean()
        
        cols = st.columns(n_clusters)
        for i, (cluster_id, row) in enumerate(cluster_analysis.iterrows()):
            with cols[i]:
                avg_aqi = row['AQI']
                st.markdown(f"#### 🏷️ 类别 {cluster_id}")
                st.write(f"**平均 AQI**: {avg_aqi:.1f}")
                
                tag = "🟢 清洁城市"
                if avg_aqi > 200: tag = "🔴 极重污染"
                elif avg_aqi > 150: tag = "🟠 重度污染"
                elif avg_aqi > 100: tag = "🟡 轻度污染"
                
                if row['PM10'] > 120 and (row['PM10'] / (row['PM2.5']+1) > 2):
                    tag += " (沙尘型)"
                
                st.caption(f"**特征**: {tag}")
                
                fig_feat = px.bar(
                    x=row.index, 
                    y=row.values,
                    color=row.index, 
                    color_discrete_sequence=px.colors.qualitative.Prism,
                    labels={'x': '污染物指标', 'y': '平均数值'}
                )
                fig_feat.update_layout(
                    showlegend=False,
                    xaxis_tickangle=0, 
                    margin=dict(l=0, r=0, t=0, b=0), 
                    height=180, 
                    xaxis_title=None,
                    yaxis_title=None
                )
                st.plotly_chart(fig_feat, use_container_width=True, config={'displayModeBar': False})
    else:

        st.error("数据不足，无法进行机器学习聚类。")

