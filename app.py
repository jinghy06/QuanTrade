import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="QuantTrade 量化交易监控台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式 - 现代深色主题
st.markdown("""
<style>
    /* 全局字体和背景 */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* 主标题 */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* 卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    /* 评估卡片 */
    .assessment-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        border-left: 5px solid;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    /* 事件警告 */
    .event-warning {
        background: linear-gradient(135deg, rgba(255,193,7,0.2) 0%, rgba(255,193,7,0.1) 100%);
        border: 1px solid rgba(255,193,7,0.5);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* 正负值颜色 */
    .positive { 
        color: #00e676; 
        font-weight: bold; 
        text-shadow: 0 0 10px rgba(0,230,118,0.3);
    }
    .negative { 
        color: #ff5252; 
        font-weight: bold; 
        text-shadow: 0 0 10px rgba(255,82,82,0.3);
    }
    .info-text { 
        color: #b0bec5; 
        font-size: 0.9rem; 
    }
    
    /* Tab样式 */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 8px; 
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] { 
        padding: 10px 20px;
        border-radius: 8px;
        background: transparent;
        color: #b0bec5;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.1);
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
    }
    
    /* 侧边栏 */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* 按钮样式 */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    
    /* 数据框样式 */
    .dataframe {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* 分割线 */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

DB_PATH = r'C:/Users/HY/PycharmProjects/QuanTrade/QuanTrade/quant_system/data/quant.db'
etf_list = ['562500.SH', '159382.SZ', '588790.SH', '159241.SZ', '588200.SH']

# ==================== 数据加载函数 ====================
@st.cache_data(ttl=300)
def load_daily_prices():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM daily_prices ORDER BY trade_date, symbol", conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    except Exception as e:
        st.warning(f"daily_prices 表读取失败: {e}")
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_signals_v3():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM strategy_signals_v3 ORDER BY trade_date, symbol", conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    except Exception as e:
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_backtest_v3():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM backtest_results_v3", conn)
    except Exception as e:
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_predictions_v3():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM etf_model_predictions ORDER BY trade_date, symbol", conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    except Exception as e:
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_features_v4():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM features_v4 ORDER BY trade_date, symbol", conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    except Exception as e:
        st.warning(f"features_v4 表读取失败: {e}")
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_predictions_v4():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM predictions_v4 ORDER BY trade_date, symbol", conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    except Exception as e:
        st.warning(f"predictions_v4 表读取失败: {e}")
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_scenario_signals_v4():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM scenario_signals_v4 ORDER BY trade_date, symbol", conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    except Exception as e:
        st.warning(f"scenario_signals_v4 表读取失败: {e}")
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_future_klines_v4():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM future_klines_v4 ORDER BY forecast_date, symbol, horizon_day", conn)
        df['forecast_date'] = pd.to_datetime(df['forecast_date'], format='mixed')
    except Exception as e:
        st.warning(f"future_klines_v4 表读取失败: {e}")
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_backtest_results_v4_r2():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM backtest_results_v4_r2", conn)
    except Exception as e:
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_backtest_daily_v4_r2():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM backtest_daily_v4_r2 ORDER BY trade_date, symbol, strategy", conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    except Exception as e:
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_backtest_results_v4():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM backtest_results_v4", conn)
    except Exception as e:
        st.warning(f"backtest_results_v4 表读取失败: {e}")
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_backtest_results_v4_tuned():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM backtest_results_v4_tuned", conn)
    except Exception as e:
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_backtest_daily_v4():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM backtest_daily_v4 ORDER BY trade_date, symbol, strategy", conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    except Exception as e:
        st.warning(f"backtest_daily_v4 表读取失败: {e}")
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_backtest_daily_v4_tuned():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM backtest_daily_v4_tuned ORDER BY trade_date, symbol, strategy", conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    except Exception as e:
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_news_sentiment_v4():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM news_sentiment_v4 ORDER BY date, symbol", conn)
        df['date'] = pd.to_datetime(df['date'], format='mixed')
    except Exception as e:
        st.warning(f"news_sentiment_v4 表读取失败: {e}")
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_news_raw_v4():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM news_raw_v4 ORDER BY date DESC", conn)
        df['date'] = pd.to_datetime(df['date'], format='mixed')
    except Exception as e:
        st.warning(f"news_raw_v4 表读取失败: {e}")
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_backtest_results_v4_r2():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM backtest_results_v4_r2", conn)
    except Exception as e:
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_backtest_daily_v4_r2():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM backtest_daily_v4_r2 ORDER BY trade_date, symbol, strategy", conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    except Exception as e:
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_backtest_results_v4_r3():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM backtest_results_v4_r3", conn)
    except Exception as e:
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_backtest_daily_v4_r3():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM backtest_daily_v4_r3 ORDER BY trade_date, symbol, strategy", conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    except Exception as e:
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_backtest_results_v5():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM backtest_results_v5", conn)
    except Exception as e:
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_backtest_daily_v5():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM backtest_daily_v5 ORDER BY trade_date, symbol, strategy", conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    except Exception as e:
        df = pd.DataFrame()
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_predictions_v5():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM predictions_v5 ORDER BY trade_date, symbol", conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
    except Exception as e:
        df = pd.DataFrame()
    conn.close()
    return df

daily_prices = load_daily_prices()
signals = load_signals_v3()
backtest = load_backtest_v3()
predictions = load_predictions_v3()
features_v4 = load_features_v4()
predictions_v4 = load_predictions_v4()
scenario_signals_v4 = load_scenario_signals_v4()
future_klines_v4 = load_future_klines_v4()
backtest_results_v4 = load_backtest_results_v4()
backtest_results_v4_tuned = load_backtest_results_v4_tuned()
backtest_daily_v4 = load_backtest_daily_v4()
backtest_daily_v4_tuned = load_backtest_daily_v4_tuned()
backtest_results_v4_r2 = load_backtest_results_v4_r2()
backtest_daily_v4_r2 = load_backtest_daily_v4_r2()
backtest_results_v4_r3 = load_backtest_results_v4_r3()
backtest_daily_v4_r3 = load_backtest_daily_v4_r3()
backtest_results_v5 = load_backtest_results_v5()
backtest_daily_v5 = load_backtest_daily_v5()
predictions_v5 = load_predictions_v5()
news_sentiment_v4 = load_news_sentiment_v4()
news_raw_v4 = load_news_raw_v4()

# ==================== 侧边栏 ====================
st.sidebar.markdown("## 📊 QuantTrade 监控台")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "导航",
    ["🏠 数据概览", "🤖 模型评估", "📈 回测分析", "🔔 信号监控", "⚙️ 策略调参",
     "🔮 情景分析", "📡 未来走势", "📰 新闻情绪", "🧪 OOD回测"]
)

selected_etf = st.sidebar.selectbox("选择ETF", etf_list)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 数据状态")
if not daily_prices.empty:
    latest_date = daily_prices['trade_date'].max()
    st.sidebar.info(f"最新数据: {latest_date.strftime('%Y-%m-%d')}")
    st.sidebar.info(f"ETF数量: {daily_prices['symbol'].nunique()} 只")
    st.sidebar.info(f"总记录: {len(daily_prices)} 条")

# ==================== 辅助函数 ====================
def get_trend_color(state):
    if state in ['强涨']:
        return '#2ca02c'
    elif state in ['温和涨']:
        return '#66bb6a'
    elif state in ['震荡']:
        return '#1f77b4'
    elif state in ['温和跌']:
        return '#ff8a65'
    elif state in ['强跌']:
        return '#d62728'
    return '#888'

def get_vol_color(state):
    if state in ['低']:
        return '#2ca02c'
    elif state in ['正常']:
        return '#1f77b4'
    elif state in ['高']:
        return '#d62728'
    return '#888'

def get_dd_color(state):
    if state in ['峰值附近']:
        return '#2ca02c'
    elif state in ['回撤']:
        return '#ff9800'
    elif state in ['修正']:
        return '#ff5722'
    elif state in ['深度回撤']:
        return '#d62728'
    return '#888'

def get_sr_color(state):
    if state in ['接近支撑']:
        return '#2ca02c'
    elif state in ['中间']:
        return '#1f77b4'
    elif state in ['接近阻力']:
        return '#d62728'
    return '#888'

def color_return(val):
    color = '#2ca02c' if val > 0 else '#d62728'
    return f'color: {color}; font-weight: bold'

# ==================== 页面1: 数据概览 ====================
if page == "🏠 数据概览":
    st.markdown('<div class="main-header">🏠 数据概览</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    if not daily_prices.empty:
        etf_data = daily_prices[daily_prices['symbol'] == selected_etf]
        if not etf_data.empty:
            latest = etf_data.iloc[-1]
            prev = etf_data.iloc[-2] if len(etf_data) > 1 else latest
            change_pct = latest['pct_change'] if not pd.isna(latest['pct_change']) else 0
            
            col1.metric("最新收盘价", f"{latest['close']:.3f}", f"{change_pct:+.2f}%")
            col2.metric("成交量", f"{latest['volume']/10000:.1f}万", f"{(latest['volume']/prev['volume']-1)*100:+.1f}%" if prev['volume'] > 0 else "N/A")
            col3.metric("成交额", f"{latest['amount']/100000000:.2f}亿")
            col4.metric("换手率", f"{latest['turnover']:.2f}%")
    
    st.markdown("---")
    
    # K线图
    st.subheader(f"📊 {selected_etf} K线走势")
    
    etf_kline = daily_prices[daily_prices['symbol'] == selected_etf].sort_values('trade_date')
    if not etf_kline.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=etf_kline['trade_date'],
            open=etf_kline['open'],
            high=etf_kline['high'],
            low=etf_kline['low'],
            close=etf_kline['close'],
            name='K线'
        )])
        
        # 添加MA5和MA20
        etf_kline['ma5'] = etf_kline['close'].rolling(5).mean()
        etf_kline['ma20'] = etf_kline['close'].rolling(20).mean()
        
        fig.add_trace(go.Scatter(x=etf_kline['trade_date'], y=etf_kline['ma5'], 
                                  mode='lines', name='MA5', line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=etf_kline['trade_date'], y=etf_kline['ma20'], 
                                  mode='lines', name='MA20', line=dict(color='blue', width=1)))
        
        fig.update_layout(
            title=f"{selected_etf} 日K线",
            xaxis_title="日期",
            yaxis_title="价格",
            height=500,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 各ETF数据量统计
    st.subheader("📋 各ETF数据统计")
    stats = []
    for etf in etf_list:
        etf_df = daily_prices[daily_prices['symbol'] == etf]
        if not etf_df.empty:
            stats.append({
                'ETF代码': etf,
                '数据条数': len(etf_df),
                '起始日期': etf_df['trade_date'].min().strftime('%Y-%m-%d'),
                '结束日期': etf_df['trade_date'].max().strftime('%Y-%m-%d'),
                '最新收盘价': etf_df['close'].iloc[-1],
                '区间涨幅': (etf_df['close'].iloc[-1] / etf_df['close'].iloc[0] - 1) * 100
            })
    
    stats_df = pd.DataFrame(stats)
    st.dataframe(stats_df, use_container_width=True)
    
    # 成交量对比
    st.subheader("📊 成交量对比")
    vol_df = daily_prices.groupby(['trade_date', 'symbol'])['volume'].sum().reset_index()
    fig_vol = px.line(vol_df, x='trade_date', y='volume', color='symbol', 
                       title="各ETF成交量走势", height=400)
    st.plotly_chart(fig_vol, use_container_width=True)

# ==================== 页面2: 模型评估 ====================
elif page == "🤖 模型评估":
    st.markdown('<div class="main-header">🤖 模型评估</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LightGBM 准确率", "75.60%", "+26.85%")
    col2.metric("RandomForest 准确率", "77.60%", "+28.85%")
    col3.metric("融合模型 准确率", "76.40%", "+27.65%")
    col4.metric("融合模型 AUC", "0.8637", "+0.3762")
    
    st.markdown("---")
    
    # 模型对比
    st.subheader("📊 模型性能对比")
    model_comparison = pd.DataFrame({
        '模型': ['LightGBM', 'RandomForest', '融合模型'],
        '准确率': [0.7560, 0.7760, 0.7640],
        '精确率': [0.5882, 0.7391, 0.6296],
        '召回率': [0.2985, 0.2537, 0.4359],
        'F1分数': [0.3960, 0.3778, 0.5152],
        'AUC': [0.8671, 0.8523, 0.8637]
    })
    
    fig = go.Figure()
    metrics = ['准确率', '精确率', '召回率', 'F1分数', 'AUC']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, model in enumerate(model_comparison['模型']):
        fig.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=model_comparison.iloc[i][metrics].values,
            marker_color=colors[i]
        ))
    
    fig.update_layout(
        barmode='group',
        title="各模型性能指标对比",
        yaxis_title="得分",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 特征重要性
    st.subheader("🔍 Top 15 特征重要性")
    feature_importance = pd.DataFrame({
        '特征': ['volume_ratio', 'return_1d', 'trend_slope_10', 'upper_shadow', 'amount',
                'macd_dea', 'amplitude', 'obv_ratio', 'macd_hist', 'turnover',
                'atr_14', 'body_pct', 'amount_ratio', 'kdj_d', 'rsi_14'],
        '重要性': [161, 148, 129, 125, 124, 120, 113, 110, 108, 107, 104, 98, 98, 98, 96]
    })
    
    fig_imp = px.bar(feature_importance, x='重要性', y='特征', orientation='h',
                      title="特征重要性排名", height=500)
    fig_imp.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # 混淆矩阵
    st.subheader("🎯 混淆矩阵 (融合模型)")
    cm_data = np.array([[81, 10], [22, 17]])
    fig_cm = px.imshow(cm_data, 
                       labels=dict(x="预测", y="实际", color="数量"),
                       x=['跌(0)', '涨(1)'],
                       y=['跌(0)', '涨(1)'],
                       text_auto=True,
                       color_continuous_scale='Blues',
                       height=400)
    fig_cm.update_layout(title="混淆矩阵")
    st.plotly_chart(fig_cm, use_container_width=True)

# ==================== 页面3: 回测分析 ====================
elif page == "📈 回测分析":
    st.markdown('<div class="main-header">📈 回测分析</div>', unsafe_allow_html=True)
    
    # 从数据库加载回测结果
    if not backtest.empty:
        backtest_summary = backtest.copy()
        backtest_summary['总收益率(%)'] = backtest_summary['total_return'] * 100
        backtest_summary['最大回撤(%)'] = backtest_summary['max_drawdown'] * 100
        backtest_summary['年化收益率(%)'] = backtest_summary['annual_return'] * 100
        backtest_summary['胜率(%)'] = backtest_summary['win_rate'] * 100
        backtest_summary['基准收益(%)'] = backtest_summary['benchmark_return'] * 100
        
        display_df = backtest_summary[['symbol', 'strategy', '总收益率(%)', '基准收益(%)', '最大回撤(%)', 'sharpe', '胜率(%)']].copy()
        display_df.columns = ['ETF', '策略', '总收益率(%)', '基准收益(%)', '最大回撤(%)', '夏普比率', '胜率(%)']
    else:
        display_df = pd.DataFrame({
            'ETF': ['159241.SZ', '159382.SZ', '562500.SH', '588200.SH', '588790.SH'] * 5,
            '策略': ['基础融合']*5 + ['趋势过滤']*5 + ['动态阈值']*5 + ['Kelly仓位']*5 + ['综合策略']*5,
            '总收益率(%)': [-1.48, 11.95, 3.38, 9.53, 0.39, -2.35, 11.95, 3.38, 9.53, 3.34,
                            -3.58, 11.23, 3.38, 11.92, 1.80, 3.41, 6.20, 7.69, 9.51, 6.98,
                            -0.36, 3.53, 0.57, 3.91, 0.03],
            '基准收益(%)': [-9.12, 13.96, 13.91, 19.21, 6.42] * 5,
            '最大回撤(%)': [-3.25, -1.32, -4.84, -0.02, -7.42, -2.72, -1.32, -4.84, -0.02, -4.86,
                          -5.32, -3.96, -4.84, -0.02, -7.42, -1.79, -3.69, -1.72, -2.79, -2.14,
                          -0.56, -0.61, -1.35, -0.02, -1.69],
            '夏普比率': [-1.24, 3.77, 1.33, 3.17, 0.20, -2.61, 3.77, 1.33, 3.17, 1.25,
                      -2.42, 2.94, 1.33, 3.75, 0.67, 2.83, 2.64, 4.89, 2.40, 2.04,
                      -1.85, 3.23, 0.70, 2.92, 0.05],
            '胜率(%)': [10.2, 8.2, 10.2, 4.1, 10.2, 6.1, 8.2, 10.2, 4.1, 10.2,
                      10.2, 10.2, 10.2, 6.1, 10.2, 18.4, 22.4, 30.6, 28.6, 16.3,
                      4.1, 8.2, 8.2, 4.1, 4.1]
        })
    
    # 策略筛选
    all_strategies = display_df['策略'].unique().tolist()
    if '基准(持有)' in all_strategies:
        all_strategies.remove('基准(持有)')
    selected_strategies = st.multiselect("选择要对比的策略", all_strategies, default=all_strategies, key='backtest_strategies')
    
    if selected_strategies:
        display_df = display_df[display_df['策略'].isin(selected_strategies)]
    
    st.subheader("📋 回测结果汇总")
    st.dataframe(display_df.style.applymap(color_return, subset=['总收益率(%)', '基准收益(%)']), use_container_width=True)
    
    # 收益率热力图
    st.subheader("🔥 策略收益率热力图")
    pivot_df = display_df.pivot(index='ETF', columns='策略', values='总收益率(%)')
    fig_heat = px.imshow(pivot_df, 
                         labels=dict(x="策略", y="ETF", color="总收益率(%)"),
                         text_auto='.2f',
                         color_continuous_scale='RdYlGn',
                         height=400)
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # 策略vs基准对比柱状图
    st.subheader("📊 策略 vs 基准收益对比")
    fig_compare = go.Figure()
    for etf in display_df['ETF'].unique():
        etf_data = display_df[display_df['ETF'] == etf]
        fig_compare.add_trace(go.Bar(
            name=f'{etf} 策略',
            x=etf_data['策略'],
            y=etf_data['总收益率(%)'],
            marker_color='#1f77b4'
        ))
        fig_compare.add_trace(go.Bar(
            name=f'{etf} 基准',
            x=etf_data['策略'],
            y=etf_data['基准收益(%)'],
            marker_color='#ff7f0e',
            opacity=0.6
        ))
    fig_compare.update_layout(barmode='group', height=500, title="策略收益 vs 基准收益")
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # 策略对比雷达图
    st.subheader("🕸️ 策略综合表现雷达图")
    
    avg_perf = display_df.groupby('策略').agg({
        '总收益率(%)': 'mean',
        '最大回撤(%)': 'mean',
        '夏普比率': 'mean'
    }).reset_index()
    
    categories = ['总收益率', '最大回撤(反向)', '夏普比率']
    fig_radar = go.Figure()
    
    for _, row in avg_perf.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['总收益率(%)'], -row['最大回撤(%)'], row['夏普比率']*2],
            theta=categories,
            fill='toself',
            name=row['策略']
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-5, 15])),
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ==================== 页面4: 信号监控 ====================
elif page == "🔔 信号监控":
    st.markdown('<div class="main-header">🔔 实时信号监控</div>', unsafe_allow_html=True)
    
    if not signals.empty:
        # 最新信号
        latest_signals = signals.groupby('symbol').last().reset_index()
        
        st.subheader("📡 最新交易信号")
        
        cols = st.columns(5)
        for i, (_, row) in enumerate(latest_signals.iterrows()):
            with cols[i % 5]:
                signal_type = "🟢 买入" if row.get('signal_combined', 0) == 1 else "🔴 观望"
                prob = row.get('fusion_prob', 0)
                kelly = row.get('kelly_position', 0)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{row['symbol']}</h4>
                    <p>信号: {signal_type}</p>
                    <p>上涨概率: <span class="{'positive' if prob > 0.5 else 'negative'}">{prob*100:.1f}%</span></p>
                    <p>Kelly仓位: {kelly*100:.1f}%</p>
                    <p>日期: {row['trade_date'].strftime('%Y-%m-%d')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 信号历史
        st.subheader("📊 信号历史")
        selected_signal_etf = st.selectbox("选择ETF查看信号历史", etf_list, key='signal_etf')
        
        signal_hist = signals[signals['symbol'] == selected_signal_etf].sort_values('trade_date')
        if not signal_hist.empty:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=('价格走势', '预测概率', '信号与仓位'),
                                vertical_spacing=0.08)
            
            # 价格
            etf_price = daily_prices[daily_prices['symbol'] == selected_signal_etf].sort_values('trade_date')
            fig.add_trace(go.Scatter(x=etf_price['trade_date'], y=etf_price['close'],
                                      mode='lines', name='收盘价', line=dict(color='blue')), row=1, col=1)
            
            # 预测概率
            fig.add_trace(go.Scatter(x=signal_hist['trade_date'], y=signal_hist['fusion_prob'],
                                      mode='lines', name='上涨概率', line=dict(color='green')), row=2, col=1)
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=2, col=1)
            
            # 信号
            fig.add_trace(go.Scatter(x=signal_hist['trade_date'], y=signal_hist['signal_combined'],
                                      mode='lines', name='交易信号', line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Scatter(x=signal_hist['trade_date'], y=signal_hist['kelly_position'],
                                      mode='lines', name='Kelly仓位', line=dict(color='purple')), row=3, col=1)
            
            fig.update_layout(height=700, showlegend=True,
                             title=f"{selected_signal_etf} 信号监控")
            st.plotly_chart(fig, use_container_width=True)
        
        # 信号统计
        st.subheader("📈 信号统计")
        signal_stats = []
        for etf in etf_list:
            etf_sig = signals[signals['symbol'] == etf]
            if not etf_sig.empty:
                buy_count = etf_sig['signal_combined'].sum()
                total = len(etf_sig)
                avg_prob = etf_sig['fusion_prob'].mean()
                avg_kelly = etf_sig['kelly_position'].mean()
                
                signal_stats.append({
                    'ETF': etf,
                    '总信号数': total,
                    '买入信号数': int(buy_count),
                    '买入比例': f"{buy_count/total*100:.1f}%",
                    '平均概率': f"{avg_prob*100:.1f}%",
                    '平均Kelly仓位': f"{avg_kelly*100:.1f}%"
                })
        
        st.dataframe(pd.DataFrame(signal_stats), use_container_width=True)
    else:
        st.warning("暂无信号数据，请先运行策略脚本生成信号。")

# ==================== 页面5: 策略调参 ====================
elif page == "⚙️ 策略调参":
    st.markdown('<div class="main-header">⚙️ 策略参数调优</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-text">
    调整以下参数，实时查看策略表现变化。注意：当前为演示模式，参数调整不会重新训练模型。
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 分类阈值参数")
        base_threshold = st.slider("基础阈值", 0.30, 0.70, 0.50, 0.01)
        bull_threshold = st.slider("牛市阈值 (MA多头排列)", 0.30, 0.70, 0.45, 0.01)
        bear_threshold = st.slider("熊市阈值 (MA空头排列)", 0.30, 0.70, 0.60, 0.01)
    
    with col2:
        st.subheader("🛡️ 风控参数")
        fee_rate = st.slider("手续费率 (%)", 0.00, 0.10, 0.01, 0.001) / 100
        slippage = st.slider("滑点 (%)", 0.00, 0.10, 0.01, 0.001) / 100
        stop_loss_atr = st.slider("ATR止损倍数", 1.0, 5.0, 2.0, 0.1)
        max_position = st.slider("最大仓位 (%)", 50, 100, 100, 5) / 100
    
    st.markdown("---")
    
    # 参数影响分析
    st.subheader("📈 参数影响分析")
    
    # 模拟不同阈值下的表现
    threshold_range = np.arange(0.35, 0.65, 0.02)
    
    if not signals.empty:
        sim_results = []
        for thresh in threshold_range:
            for etf in etf_list:
                etf_sig = signals[signals['symbol'] == etf].copy()
                if not etf_sig.empty:
                    etf_sig['sim_signal'] = (etf_sig['fusion_prob'] > thresh).astype(int)
                    buy_rate = etf_sig['sim_signal'].mean()
                    
                    # 简单模拟收益
                    actual = etf_sig['target_next_day_return'].fillna(0)
                    sim_returns = actual * etf_sig['sim_signal']
                    total_ret = (1 + sim_returns).prod() - 1
                    
                    sim_results.append({
                        '阈值': thresh,
                        'ETF': etf,
                        '买入比例': buy_rate * 100,
                        '模拟总收益': total_ret * 100
                    })
        
        sim_df = pd.DataFrame(sim_results)
        
        fig = px.line(sim_df, x='阈值', y='模拟总收益', color='ETF',
                      title="不同阈值下的模拟总收益", height=450)
        fig.add_vline(x=base_threshold, line_dash="dash", line_color="red",
                      annotation_text="当前阈值")
        st.plotly_chart(fig, use_container_width=True)
        
        # 买入比例 vs 阈值
        fig2 = px.line(sim_df, x='阈值', y='买入比例', color='ETF',
                       title="不同阈值下的买入比例", height=450)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # 策略配置导出
    st.subheader("💾 策略配置")
    config = {
        'base_threshold': base_threshold,
        'bull_threshold': bull_threshold,
        'bear_threshold': bear_threshold,
        'fee_rate': fee_rate,
        'slippage': slippage,
        'stop_loss_atr': stop_loss_atr,
        'max_position': max_position,
        'generated_at': datetime.now().isoformat()
    }
    
    st.json(config)
    
    if st.button("📥 导出配置"):
        st.download_button(
            label="下载JSON配置",
            data=pd.Series(config).to_json(),
            file_name=f"strategy_config_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

# ==================== 页面6: 情景分析 ====================
elif page == "🔮 情景分析":
    st.markdown('<div class="main-header">🔮 情景分析</div>', unsafe_allow_html=True)
    
    selected_etf_scenario = st.selectbox("选择ETF", etf_list, key='scenario_etf')
    
    # 获取最新数据
    latest_features = features_v4[features_v4['symbol'] == selected_etf_scenario].sort_values('trade_date')
    latest_pred = predictions_v4[predictions_v4['symbol'] == selected_etf_scenario].sort_values('trade_date')
    latest_signal = scenario_signals_v4[scenario_signals_v4['symbol'] == selected_etf_scenario].sort_values('trade_date')
    
    if latest_features.empty or latest_pred.empty or latest_signal.empty:
        st.warning("情景分析数据不足，请确保 features_v4、predictions_v4 和 scenario_signals_v4 表中有数据。")
    else:
        feat = latest_features.iloc[-1]
        pred = latest_pred.iloc[-1]
        sig = latest_signal.iloc[-1]
        
        # Assessment状态卡片
        st.subheader("📊 Assessment状态")
        c1, c2, c3, c4 = st.columns(4)
        
        trend_state = feat.get('trend_state', '未知')
        vol_state = feat.get('vol_state', '未知')
        drawdown_state = feat.get('drawdown_state', '未知')
        sr_state = feat.get('sr_state', '未知')
        event_state = int(feat.get('event_state', 0))
        
        with c1:
            tc = get_trend_color(trend_state)
            st.markdown(f"""
            <div class="assessment-card" style="border-left-color: {tc};">
                <h4 style="margin:0;color:#666;">趋势状态</h4>
                <h2 style="margin:0.5rem 0;color:{tc};">{trend_state}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            vc = get_vol_color(vol_state)
            st.markdown(f"""
            <div class="assessment-card" style="border-left-color: {vc};">
                <h4 style="margin:0;color:#666;">波动状态</h4>
                <h2 style="margin:0.5rem 0;color:{vc};">{vol_state}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with c3:
            dc = get_dd_color(drawdown_state)
            st.markdown(f"""
            <div class="assessment-card" style="border-left-color: {dc};">
                <h4 style="margin:0;color:#666;">回撤状态</h4>
                <h2 style="margin:0.5rem 0;color:{dc};">{drawdown_state}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with c4:
            sc = get_sr_color(sr_state)
            st.markdown(f"""
            <div class="assessment-card" style="border-left-color: {sc};">
                <h4 style="margin:0;color:#666;">支撑阻力</h4>
                <h2 style="margin:0.5rem 0;color:{sc};">{sr_state}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # 事件标记
        if event_state == 1:
            st.markdown('<div class="event-warning">⚠️ 检测到异常波动事件</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 三情景概率条形图
        st.subheader("📊 三情景概率")
        
        rf_probs = [pred.get('rf_adverse_proba', 0), pred.get('rf_base_proba', 0), pred.get('rf_favorable_proba', 0)]
        fusion_probs = [pred.get('fusion_adverse_proba', 0), pred.get('fusion_base_proba', 0), pred.get('fusion_favorable_proba', 0)]
        scenarios = ['P(adverse)', 'P(base)', 'P(favorable)']
        
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Bar(
            name='RandomForest',
            x=scenarios,
            y=rf_probs,
            marker_color=['#d62728', '#1f77b4', '#2ca02c']
        ))
        fig_prob.add_trace(go.Bar(
            name='Fusion',
            x=scenarios,
            y=fusion_probs,
            marker_color=['#d62728', '#1f77b4', '#2ca02c'],
            opacity=0.6
        ))
        fig_prob.update_layout(
            barmode='group',
            title="三情景概率对比 (RF vs Fusion)",
            yaxis_title="概率",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # 推荐仓位仪表盘
        st.subheader("🎯 推荐仓位")
        
        position_size = float(sig.get('position_size', 0)) if not pd.isna(sig.get('position_size')) else 0
        scenario_decision = sig.get('scenario_decision', '未知')
        signal_direction = sig.get('signal_direction', '未知')
        
        col_g1, col_g2, col_g3 = st.columns([1, 1, 1])
        
        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=position_size * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "推荐仓位 (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#2ca02c' if position_size > 0.3 else '#1f77b4' if position_size > 0 else '#d62728'},
                    'steps': [
                        {'range': [0, 33], 'color': '#ffebee'},
                        {'range': [33, 66], 'color': '#e3f2fd'},
                        {'range': [66, 100], 'color': '#e8f5e9'}
                    ],
                    'threshold': {
                        'line': {'color': 'black', 'width': 4},
                        'thickness': 0.75,
                        'value': position_size * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col_g2:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;margin-top:2rem;">
                <h4 style="color:#666;">情景决策</h4>
                <h2 style="color:#1f77b4;">{scenario_decision}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col_g3:
            sig_color = '#2ca02c' if str(signal_direction) == '1' or str(signal_direction).lower() == 'buy' else '#d62728'
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;margin-top:2rem;">
                <h4 style="color:#666;">信号方向</h4>
                <h2 style="color:{sig_color};">{signal_direction}</h2>
            </div>
            """, unsafe_allow_html=True)

# ==================== 页面7: 未来走势 ====================
elif page == "📡 未来走势":
    st.markdown('<div class="main-header">📡 未来走势预测</div>', unsafe_allow_html=True)
    
    selected_etf_future = st.selectbox("选择ETF", etf_list, key='future_etf')
    
    # 获取历史K线（最近60日）
    hist_kline = daily_prices[daily_prices['symbol'] == selected_etf_future].sort_values('trade_date').tail(60)
    
    # 获取未来预测数据
    future_data = future_klines_v4[future_klines_v4['symbol'] == selected_etf_future].sort_values(['forecast_date', 'horizon_day'])
    
    if hist_kline.empty:
        st.warning("历史K线数据不足。")
    elif future_data.empty:
        st.warning("未来预测数据不足，请确保 future_klines_v4 表中有数据。")
    else:
        # 获取最新收盘价作为基准
        latest_close = hist_kline['close'].iloc[-1]
        latest_date = hist_kline['trade_date'].iloc[-1]
        
        # 取最新forecast_date的数据
        latest_forecast_date = future_data['forecast_date'].max()
        future_subset = future_data[future_data['forecast_date'] == latest_forecast_date].sort_values('horizon_day')
        
        # 构建未来日期序列（交易日，假设20个交易日约28个自然日）
        future_dates = pd.date_range(start=latest_date + pd.Timedelta(days=1), periods=len(future_subset), freq='B')
        
        # 转换为实际价格（乘以最新收盘价）
        future_subset = future_subset.copy()
        future_subset['median_price'] = future_subset['median_close'] * latest_close
        future_subset['p10_price'] = future_subset['p10_close'] * latest_close
        future_subset['p90_price'] = future_subset['p90_close'] * latest_close
        future_subset['forecast_date_abs'] = future_dates[:len(future_subset)]
        
        # 历史K线 + 未来预测区间
        st.subheader("📈 历史K线 + 未来预测区间")
        
        fig_future = go.Figure()
        
        # 历史K线
        fig_future.add_trace(go.Candlestick(
            x=hist_kline['trade_date'],
            open=hist_kline['open'],
            high=hist_kline['high'],
            low=hist_kline['low'],
            close=hist_kline['close'],
            name='历史K线'
        ))
        
        # 未来中位数路径（虚线）
        fig_future.add_trace(go.Scatter(
            x=future_subset['forecast_date_abs'],
            y=future_subset['median_price'],
            mode='lines',
            name='预测中位数',
            line=dict(color='#1f77b4', width=2, dash='dash')
        ))
        
        # P10/P90置信区间（半透明填充）
        fig_future.add_trace(go.Scatter(
            x=list(future_subset['forecast_date_abs']) + list(future_subset['forecast_date_abs'][::-1]),
            y=list(future_subset['p90_price']) + list(future_subset['p10_price'][::-1]),
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='P10-P90置信区间'
        ))
        
        fig_future.update_layout(
            title=f"{selected_etf_future} 历史走势 + 20日Monte Carlo预测",
            xaxis_title="日期",
            yaxis_title="价格",
            height=550,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig_future, use_container_width=True)
        
        # Monte Carlo路径密度图（用面积图展示置信区间）
        st.subheader("📊 Monte Carlo预测区间")
        
        fig_mc = go.Figure()
        
        fig_mc.add_trace(go.Scatter(
            x=future_subset['forecast_date_abs'],
            y=future_subset['p90_price'],
            mode='lines',
            name='P90',
            line=dict(color='#2ca02c', width=1)
        ))
        
        fig_mc.add_trace(go.Scatter(
            x=future_subset['forecast_date_abs'],
            y=future_subset['median_price'],
            mode='lines',
            name='中位数',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig_mc.add_trace(go.Scatter(
            x=future_subset['forecast_date_abs'],
            y=future_subset['p10_price'],
            mode='lines',
            name='P10',
            line=dict(color='#d62728', width=1)
        ))
        
        # 填充区域
        fig_mc.add_trace(go.Scatter(
            x=list(future_subset['forecast_date_abs']) + list(future_subset['forecast_date_abs'][::-1]),
            y=list(future_subset['p90_price']) + list(future_subset['p10_price'][::-1]),
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='置信区间',
            showlegend=False
        ))
        
        fig_mc.update_layout(
            title="预测价格区间 (P10 / 中位数 / P90)",
            xaxis_title="日期",
            yaxis_title="价格",
            height=400
        )
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # 预测参数卡片
        st.subheader("📋 预测参数")
        
        scenario_decision_f = future_subset['scenario_decision'].iloc[0] if 'scenario_decision' in future_subset.columns else '未知'
        
        # 计算20日预期收益区间
        p10_return = (future_subset['p10_price'].iloc[-1] / latest_close - 1) * 100
        p90_return = (future_subset['p90_price'].iloc[-1] / latest_close - 1) * 100
        median_return = (future_subset['median_price'].iloc[-1] / latest_close - 1) * 100
        
        # 漂移调整推断
        drift = "中性"
        if scenario_decision_f == 'adverse':
            drift = "负漂移"
        elif scenario_decision_f == 'favorable':
            drift = "正漂移"
        elif scenario_decision_f == 'base':
            drift = "基准漂移"
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("情景决策", str(scenario_decision_f))
        c2.metric("漂移调整", drift)
        c3.metric("20日预期收益(P10)", f"{p10_return:.1f}%")
        c4.metric("20日预期收益(P90)", f"{p90_return:.1f}%")
        
        st.info(f"20日中位数预期收益: **{median_return:.1f}%** | 当前基准价: **{latest_close:.3f}**")

# ==================== 页面8: 新闻情绪 ====================
elif page == "📰 新闻情绪":
    st.markdown('<div class="main-header">📰 新闻情绪监控</div>', unsafe_allow_html=True)
    
    selected_etf_news = st.selectbox("选择ETF", etf_list, key='news_etf')
    
    # 新闻时间线
    st.subheader("📰 最近7天新闻")
    
    if not news_raw_v4.empty:
        etf_news = news_raw_v4[news_raw_v4['symbol'] == selected_etf_news].copy()
        if not etf_news.empty:
            # 最近7天
            cutoff = datetime.now() - timedelta(days=7)
            recent_news = etf_news[etf_news['date'] >= cutoff].sort_values('date', ascending=False)
            
            if not recent_news.empty:
                for _, row in recent_news.head(20).iterrows():
                    sentiment = row.get('sentiment_score', 0)
                    is_major = int(row.get('is_major_event', 0))
                    
                    if sentiment > 0.1:
                        color = '#2ca02c'
                        label = '正向'
                    elif sentiment < -0.1:
                        color = '#d62728'
                        label = '负向'
                    else:
                        color = '#888'
                        label = '中性'
                    
                    major_icon = '🔴 ' if is_major == 1 else ''
                    
                    st.markdown(f"""
                    <div style="border-left: 4px solid {color}; padding-left: 10px; margin: 8px 0;">
                        <span style="color:#888;font-size:0.85rem;">{row['date'].strftime('%Y-%m-%d')}</span>
                        {major_icon}<b>{row.get('title', '无标题')}</b>
                        <span style="color:{color};font-size:0.85rem;">[{label}]</span>
                        <span style="color:#666;font-size:0.8rem;">score: {sentiment:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("最近7天无新闻数据。")
        else:
            st.info(f"暂无 {selected_etf_news} 的新闻数据。")
    else:
        st.warning("news_raw_v4 表数据为空。")
    
    st.markdown("---")
    
    # 情绪曲线
    st.subheader("📊 情绪变化曲线")
    
    if not news_sentiment_v4.empty:
        etf_sentiment = news_sentiment_v4[news_sentiment_v4['symbol'] == selected_etf_news].sort_values('date')
        
        if not etf_sentiment.empty:
            # 取最近30天
            cutoff = datetime.now() - timedelta(days=30)
            etf_sentiment = etf_sentiment[etf_sentiment['date'] >= cutoff]
            
            fig_sent = go.Figure()
            fig_sent.add_trace(go.Scatter(
                x=etf_sentiment['date'],
                y=etf_sentiment['sentiment_1d'],
                mode='lines+markers',
                name='1日情绪',
                line=dict(color='#d62728')
            ))
            fig_sent.add_trace(go.Scatter(
                x=etf_sentiment['date'],
                y=etf_sentiment['sentiment_3d'],
                mode='lines+markers',
                name='3日情绪',
                line=dict(color='#1f77b4')
            ))
            fig_sent.add_trace(go.Scatter(
                x=etf_sentiment['date'],
                y=etf_sentiment['sentiment_7d'],
                mode='lines+markers',
                name='7日情绪',
                line=dict(color='#2ca02c')
            ))
            
            fig_sent.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_sent.update_layout(
                title=f"{selected_etf_news} 近30日情绪变化",
                xaxis_title="日期",
                yaxis_title="情绪得分",
                height=400
            )
            st.plotly_chart(fig_sent, use_container_width=True)
        else:
            st.info("情绪曲线数据不足。")
    else:
        st.warning("news_sentiment_v4 表数据为空。")
    
    st.markdown("---")
    
    # 情绪统计卡片
    st.subheader("📋 情绪统计")
    
    if not news_sentiment_v4.empty:
        etf_sentiment = news_sentiment_v4[news_sentiment_v4['symbol'] == selected_etf_news].sort_values('date')
        
        if not etf_sentiment.empty:
            latest_sent = etf_sentiment.iloc[-1]
            
            c1, c2, c3, c4 = st.columns(4)
            
            s1d = latest_sent.get('sentiment_1d', 0)
            s3d = latest_sent.get('sentiment_3d', 0)
            s7d = latest_sent.get('sentiment_7d', 0)
            major_3d = int(latest_sent.get('major_events_3d', 0))
            latest_title = latest_sent.get('latest_title', '无')
            
            c1.metric("近1日情绪", f"{s1d:.3f}")
            c2.metric("近3日情绪", f"{s3d:.3f}")
            c3.metric("近7日情绪", f"{s7d:.3f}")
            c4.metric("近3日重大事件", f"{major_3d} 件")
            
            st.markdown(f"**最新新闻标题**: {latest_title}")
        else:
            st.info("暂无该ETF的情绪统计数据。")
    else:
        st.warning("news_sentiment_v4 表数据为空。")

# ==================== 页面9: OOD回测 ====================
elif page == "🧪 OOD回测":
    st.markdown('<div class="main-header">🧪 OOD回测 (Out-of-Distribution)</div>', unsafe_allow_html=True)
    
    # 训练/测试集划分说明
    st.subheader("📊 训练/测试集划分")
    
    split_data = pd.DataFrame({
        '类型': ['训练集', '训练集', '训练集', '训练集', '测试集', '测试集', '测试集', '测试集', '测试集'],
        '标的': ['000002.SZ', '000333.SZ', '000568.SZ', '000651.SZ',
                '562500.SH', '588200.SH', '588790.SH', '159382.SZ', '159241.SZ'],
        '数据条数': [585, 585, 585, 585, '-', '-', '-', '-', '-'],
        '说明': ['大盘股', '大盘股', '大盘股', '大盘股',
                 'ETF(完全未见过)', 'ETF(完全未见过)', 'ETF(完全未见过)', 'ETF(完全未见过)', 'ETF(完全未见过)']
    })
    st.dataframe(split_data, use_container_width=True)
    
    st.info("训练集: 4只大盘股 (000002.SZ, 000333.SZ, 000568.SZ, 000651.SZ) — 各585条 | 测试集: 5只ETF — 完全未见过")
    
    st.markdown("---")
    
    # 策略对比表格
    st.subheader("📋 策略对比 (调优版)")
    
    # 优先展示 tuned 结果，若不存在则 fallback 到原始 v4
    bt_source = backtest_results_v4_tuned if not backtest_results_v4_tuned.empty else backtest_results_v4
    bt_daily_source = backtest_daily_v4_tuned if not backtest_daily_v4_tuned.empty else backtest_daily_v4
    
    if not bt_source.empty:
        bt_df = bt_source.copy()
        bt_df['总收益率(%)'] = bt_df['total_return'] * 100
        bt_df['年化收益率(%)'] = bt_df['annual_return'] * 100
        bt_df['最大回撤(%)'] = bt_df['max_drawdown'] * 100
        bt_df['胜率(%)'] = bt_df['win_rate'] * 100
        bt_df['基准收益(%)'] = bt_df['benchmark_return'] * 100
        
        display_bt = bt_df[['symbol', 'strategy', '总收益率(%)', '年化收益率(%)', '最大回撤(%)', 'sharpe', '胜率(%)']].copy()
        display_bt.columns = ['ETF', '策略', '总收益率(%)', '年化收益率(%)', '最大回撤(%)', '夏普比率', '胜率(%)']
        
        st.dataframe(display_bt.style.applymap(color_return, subset=['总收益率(%)', '年化收益率(%)']), use_container_width=True)
        
        # 策略筛选
        selected_etf_bt = st.selectbox("选择ETF查看累计收益曲线", etf_list, key='ood_etf')
        
        # 累计收益曲线
        st.subheader("📈 累计收益曲线")
        
        if not bt_daily_source.empty:
            daily_bt = bt_daily_source[bt_daily_source['symbol'] == selected_etf_bt].sort_values(['strategy', 'trade_date'])
            
            if not daily_bt.empty:
                fig_cum = go.Figure()
                
                strategies = daily_bt['strategy'].unique()
                colors_map = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                
                for i, strat in enumerate(strategies):
                    strat_data = daily_bt[daily_bt['strategy'] == strat]
                    fig_cum.add_trace(go.Scatter(
                        x=strat_data['trade_date'],
                        y=strat_data['nav'],
                        mode='lines',
                        name=strat,
                        line=dict(color=colors_map[i % len(colors_map)], width=2)
                    ))
                
                fig_cum.update_layout(
                    title=f"{selected_etf_bt} 各策略累计净值曲线",
                    xaxis_title="日期",
                    yaxis_title="净值",
                    height=500
                )
                st.plotly_chart(fig_cum, use_container_width=True)
            else:
                st.info(f"暂无 {selected_etf_bt} 的每日回测数据。")
        else:
            st.warning("backtest_daily_v4_tuned 表数据为空。")
        
        # 策略对比雷达图
        st.subheader("🕸️ 策略平均表现雷达图")
        
        avg_perf = display_bt.groupby('策略').agg({
            '总收益率(%)': 'mean',
            '最大回撤(%)': 'mean',
            '夏普比率': 'mean',
            '胜率(%)': 'mean'
        }).reset_index()
        
        categories = ['总收益率', '最大回撤(反向)', '夏普比率', '胜率']
        fig_radar_ood = go.Figure()
        
        for _, row in avg_perf.iterrows():
            fig_radar_ood.add_trace(go.Scatterpolar(
                r=[row['总收益率(%)'], -row['最大回撤(%)'], row['夏普比率']*2, row['胜率(%)']/10],
                theta=categories,
                fill='toself',
                name=row['策略']
            ))
        
        fig_radar_ood.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            height=500
        )
        st.plotly_chart(fig_radar_ood, use_container_width=True)
        
    else:
        st.warning("backtest_results_v4_tuned 表数据为空，无法展示策略对比。")
    
    # Round 2 模型优化结果展示
    st.markdown("---")
    st.subheader("🔬 Round 2 模型优化结果")
    st.markdown("<p style='color:#666;'>保留全部60特征 + 手动随机过采样 + 增加树数量(RF 500, LGBM 1000+早停)</p>", unsafe_allow_html=True)
    
    if not backtest_results_v4_r2.empty:
        bt_r2_df = backtest_results_v4_r2.copy()
        bt_r2_df['总收益率(%)'] = bt_r2_df['total_return'] * 100
        bt_r2_df['年化收益率(%)'] = bt_r2_df['annual_return'] * 100
        bt_r2_df['最大回撤(%)'] = bt_r2_df['max_drawdown'] * 100
        bt_r2_df['胜率(%)'] = bt_r2_df['win_rate'] * 100
        bt_r2_df['基准收益(%)'] = bt_r2_df['benchmark_return'] * 100
        
        display_r2 = bt_r2_df[['symbol', 'strategy', '总收益率(%)', '年化收益率(%)', '最大回撤(%)', 'sharpe', '胜率(%)']].copy()
        display_r2.columns = ['ETF', '策略', '总收益率(%)', '年化收益率(%)', '最大回撤(%)', '夏普比率', '胜率(%)']
        
        st.dataframe(display_r2.style.applymap(color_return, subset=['总收益率(%)', '年化收益率(%)']), use_container_width=True)
        
        # Round 2 累计收益曲线
        st.subheader("📈 Round 2 累计收益曲线")
        
        if not backtest_daily_v4_r2.empty:
            daily_r2 = backtest_daily_v4_r2[backtest_daily_v4_r2['symbol'] == selected_etf_bt].sort_values(['strategy', 'trade_date'])
            
            if not daily_r2.empty:
                fig_cum_r2 = go.Figure()
                
                strategies_r2 = daily_r2['strategy'].unique()
                colors_map_r2 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                
                for i, strat in enumerate(strategies_r2):
                    strat_data = daily_r2[daily_r2['strategy'] == strat]
                    fig_cum_r2.add_trace(go.Scatter(
                        x=strat_data['trade_date'],
                        y=strat_data['nav'],
                        mode='lines',
                        name=strat,
                        line=dict(color=colors_map_r2[i % len(colors_map_r2)], width=2)
                    ))
                
                fig_cum_r2.update_layout(
                    title=f"{selected_etf_bt} Round 2 各策略累计净值曲线",
                    xaxis_title="日期",
                    yaxis_title="净值",
                    height=500
                )
                st.plotly_chart(fig_cum_r2, use_container_width=True)
            else:
                st.info(f"暂无 {selected_etf_bt} 的 Round 2 每日回测数据。")
        else:
            st.warning("backtest_daily_v4_r2 表数据为空。")
        
        # Round 1 vs Round 2 对比
        st.subheader("⚖️ Round 1 vs Round 2 策略平均收益对比")
        
        if not bt_source.empty:
            avg_r1 = display_bt.groupby('策略')['总收益率(%)'].mean().reset_index()
            avg_r1.columns = ['策略', 'Round 1 平均收益(%)']
            
            avg_r2 = display_r2.groupby('策略')['总收益率(%)'].mean().reset_index()
            avg_r2.columns = ['策略', 'Round 2 平均收益(%)']
            
            compare_df = pd.merge(avg_r1, avg_r2, on='策略', how='outer').fillna(0)
            
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                x=compare_df['策略'],
                y=compare_df['Round 1 平均收益(%)'],
                name='Round 1',
                marker_color='#1f77b4'
            ))
            fig_compare.add_trace(go.Bar(
                x=compare_df['策略'],
                y=compare_df['Round 2 平均收益(%)'],
                name='Round 2',
                marker_color='#ff7f0e'
            ))
            
            fig_compare.update_layout(
                barmode='group',
                title="策略平均收益对比: Round 1 vs Round 2",
                xaxis_title="策略",
                yaxis_title="平均总收益率(%)",
                height=400
            )
            st.plotly_chart(fig_compare, use_container_width=True)
    else:
        st.info("Round 2 回测数据尚未生成。运行 `python scenario_backtest_v4_r2.py` 生成。")
    
    # ==================== Round 3: 趋势跟踪+股灾预判 ====================
    st.markdown("---")
    st.markdown('<div class="main-header">🚀 Round 3: 趋势跟踪加仓 + 股灾预判</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
        <h4 style="margin:0 0 0.5rem 0;">策略核心创新</h4>
        <p style="margin:0.3rem 0;"><b>1. 趋势跟踪动态加仓:</b> 120日收益>15% + 均线多头排列 → 每次加仓20%，最高100%</p>
        <p style="margin:0.3rem 0;"><b>2. 股灾多指标预警:</b> 波动率突增(ATR>3x) / 快速回撤(>20%) / 流动性枯竭(<30%) / 短期暴跌(>20%)</p>
        <p style="margin:0.3rem 0;"><b>3. 渐进式仓位管理:</b> 三情景基础仓位(adverse=60%, base=90%, favorable=100%) + 动态调整</p>
        <p style="margin:0.3rem 0; font-size:0.85rem; opacity:0.9;">历史股灾特征: 2007年6124点PE>60倍跌73% | 2015年5178点杠杆破裂跌45%</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not backtest_results_v4_r3.empty:
        bt_r3_df = backtest_results_v4_r3.copy()
        bt_r3_df['总收益率(%)'] = bt_r3_df['total_return'] * 100
        bt_r3_df['年化收益率(%)'] = bt_r3_df['annual_return'] * 100
        bt_r3_df['最大回撤(%)'] = bt_r3_df['max_drawdown'] * 100
        bt_r3_df['胜率(%)'] = bt_r3_df['win_rate'] * 100
        bt_r3_df['基准收益(%)'] = bt_r3_df['benchmark_return'] * 100
        bt_r3_df['超额收益(%)'] = bt_r3_df['excess_return'] * 100
        
        display_r3 = bt_r3_df[['symbol', 'strategy', '总收益率(%)', '年化收益率(%)', '最大回撤(%)', 
                                'sharpe', '胜率(%)', '超额收益(%)', 'avg_position', 'alert_days', 'trend_add_days']].copy()
        display_r3.columns = ['ETF', '策略', '总收益率(%)', '年化收益率(%)', '最大回撤(%)', 
                               '夏普比率', '胜率(%)', '超额收益(%)', '平均仓位', '预警天数', '加仓天数']
        display_r3['平均仓位'] = display_r3['平均仓位'] * 100
        
        st.dataframe(display_r3.style.applymap(color_return, subset=['总收益率(%)', '年化收益率(%)', '超额收益(%)']), 
                     use_container_width=True)
        
        # Round 3 累计收益曲线
        st.subheader("📈 Round 3 累计收益曲线")
        
        if not backtest_daily_v4_r3.empty:
            daily_r3 = backtest_daily_v4_r3[backtest_daily_v4_r3['symbol'] == selected_etf_bt].sort_values('trade_date')
            
            if not daily_r3.empty:
                fig_cum_r3 = go.Figure()
                
                fig_cum_r3.add_trace(go.Scatter(
                    x=daily_r3['trade_date'],
                    y=daily_r3['nav'],
                    mode='lines',
                    name='趋势跟踪+股灾预判',
                    line=dict(color='#e74c3c', width=2.5)
                ))
                
                # 添加基准线 (买入持有)
                if not daily_r3.empty:
                    first_close = daily_r3['close'].iloc[0]
                    benchmark_nav = daily_r3['close'] / first_close
                    fig_cum_r3.add_trace(go.Scatter(
                        x=daily_r3['trade_date'],
                        y=benchmark_nav,
                        mode='lines',
                        name='买入持有基准',
                        line=dict(color='#95a5a6', width=1.5, dash='dash')
                    ))
                
                fig_cum_r3.update_layout(
                    title=f"{selected_etf_bt} Round 3 趋势跟踪+股灾预判 净值曲线",
                    xaxis_title="日期",
                    yaxis_title="净值",
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_cum_r3, use_container_width=True)
                
                # 仓位变化曲线
                st.subheader("📊 仓位变化与预警状态")
                
                fig_pos = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                         row_heights=[0.6, 0.4], vertical_spacing=0.08)
                
                # 仓位曲线
                fig_pos.add_trace(go.Scatter(
                    x=daily_r3['trade_date'],
                    y=daily_r3['position'] * 100,
                    mode='lines',
                    name='实际仓位(%)',
                    line=dict(color='#3498db', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(52, 152, 219, 0.2)'
                ), row=1, col=1)
                
                fig_pos.add_trace(go.Scatter(
                    x=daily_r3['trade_date'],
                    y=daily_r3['base_position'] * 100,
                    mode='lines',
                    name='基础仓位(%)',
                    line=dict(color='#9b59b6', width=1.5, dash='dot')
                ), row=1, col=1)
                
                # 预警指标数
                colors_alert = ['#2ecc71' if a == 0 else '#f39c12' if a == 1 else '#e67e22' if a == 2 else '#e74c3c' 
                               for a in daily_r3['alerts']]
                fig_pos.add_trace(go.Bar(
                    x=daily_r3['trade_date'],
                    y=daily_r3['alerts'],
                    name='预警指标数',
                    marker_color=colors_alert,
                    opacity=0.7
                ), row=2, col=1)
                
                fig_pos.update_layout(
                    title=f"{selected_etf_bt} 仓位管理与股灾预警",
                    height=600,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig_pos.update_yaxes(title_text="仓位(%)", row=1, col=1)
                fig_pos.update_yaxes(title_text="预警指标数", row=2, col=1)
                
                st.plotly_chart(fig_pos, use_container_width=True)
            else:
                st.info(f"暂无 {selected_etf_bt} 的 Round 3 每日回测数据。")
        else:
            st.warning("backtest_daily_v4_r3 表数据为空。")
        
        # 三版本对比
        st.subheader("⚖️ 三版本策略平均收益对比")
        
        compare_data = []
        
        if not bt_source.empty:
            avg_r1 = display_bt.groupby('策略')['总收益率(%)'].mean().reset_index()
            for _, row in avg_r1.iterrows():
                compare_data.append({'策略': row['策略'], '版本': 'Round 1 (原始)', '平均收益(%)': row['总收益率(%)']})
        
        if not backtest_results_v4_r2.empty:
            avg_r2 = display_r2.groupby('策略')['总收益率(%)'].mean().reset_index()
            for _, row in avg_r2.iterrows():
                compare_data.append({'策略': row['策略'], '版本': 'Round 2 (优化)', '平均收益(%)': row['总收益率(%)']})
        
        avg_r3 = display_r3.groupby('策略')['总收益率(%)'].mean().reset_index()
        for _, row in avg_r3.iterrows():
            compare_data.append({'策略': row['策略'], '版本': 'Round 3 (趋势+预警)', '平均收益(%)': row['总收益率(%)']})
        
        if compare_data:
            compare_all = pd.DataFrame(compare_data)
            
            fig_compare_all = px.bar(compare_all, x='策略', y='平均收益(%)', color='版本',
                                      barmode='group', height=450,
                                      color_discrete_map={
                                          'Round 1 (原始)': '#3498db',
                                          'Round 2 (优化)': '#f39c12',
                                          'Round 3 (趋势+预警)': '#e74c3c'
                                      })
            fig_compare_all.update_layout(
                title="策略平均收益对比: Round 1 vs Round 2 vs Round 3",
                xaxis_title="策略",
                yaxis_title="平均总收益率(%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_compare_all, use_container_width=True)
    else:
        st.info("Round 3 回测数据尚未生成。运行 `python backtest_v4_r3_trend_crash.py` 生成。")
    
    # ==================== v5: 20年长周期模型 ====================
    st.markdown("---")
    st.markdown('<div class="main-header">🌍 v5: 20年长周期模型 (2005-2025)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                color: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
        <h4 style="margin:0 0 0.5rem 0;">v5 核心升级</h4>
        <p style="margin:0.3rem 0;"><b>训练集扩展:</b> 22只2005年前上市大盘股 × 20年 = <b>107,248条</b> (vs v4的8,190条，<b>13倍提升</b>)</p>
        <p style="margin:0.3rem 0;"><b>时间划分:</b> 训练2005-2022 (91,474条) | 验证2023 (5,324条) | 测试2024-2025 (10,450条)</p>
        <p style="margin:0.3rem 0;"><b>模型准确率:</b> Fusion 3-scenario <b>56.74%</b> (vs v4 Round 2 的 46.3%)</p>
        <p style="margin:0.3rem 0;"><b>覆盖行业:</b> 金融/能源/消费/地产/医药/科技/制造/航空 (8大行业)</p>
        <p style="margin:0.3rem 0; font-size:0.85rem; opacity:0.9;">经历完整牛熊周期: 2007年牛市/2008年金融危机/2015年杠杆牛/2018年贸易战/2020年疫情/2024年震荡市</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not backtest_results_v5.empty:
        bt_v5_df = backtest_results_v5.copy()
        bt_v5_df['总收益率(%)'] = bt_v5_df['total_return'] * 100
        bt_v5_df['年化收益率(%)'] = bt_v5_df['annual_return'] * 100
        bt_v5_df['最大回撤(%)'] = bt_v5_df['max_drawdown'] * 100
        bt_v5_df['胜率(%)'] = bt_v5_df['win_rate'] * 100
        bt_v5_df['基准收益(%)'] = bt_v5_df['benchmark_return'] * 100
        bt_v5_df['超额收益(%)'] = bt_v5_df['excess_return'] * 100
        
        display_v5 = bt_v5_df[['symbol', 'strategy', '总收益率(%)', '年化收益率(%)', '最大回撤(%)', 
                                'sharpe', '胜率(%)', '超额收益(%)', 'avg_position', 'alert_days']].copy()
        display_v5.columns = ['股票', '策略', '总收益率(%)', '年化收益率(%)', '最大回撤(%)', 
                               '夏普比率', '胜率(%)', '超额收益(%)', '平均仓位', '预警天数']
        display_v5['平均仓位'] = display_v5['平均仓位'] * 100
        
        st.dataframe(display_v5.style.applymap(color_return, subset=['总收益率(%)', '年化收益率(%)', '超额收益(%)']), 
                     use_container_width=True)
        
        # v5 平均表现
        avg_v5_return = bt_v5_df['total_return'].mean() * 100
        avg_v5_annual = bt_v5_df['annual_return'].mean() * 100
        avg_v5_dd = bt_v5_df['max_drawdown'].mean() * 100
        avg_v5_sharpe = bt_v5_df['sharpe'].mean()
        avg_v5_excess = bt_v5_df['excess_return'].mean() * 100
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("平均总收益", f"{avg_v5_return:.2f}%")
        c2.metric("平均年化", f"{avg_v5_annual:.2f}%")
        c3.metric("平均回撤", f"{avg_v5_dd:.2f}%")
        c4.metric("平均夏普", f"{avg_v5_sharpe:.2f}")
        c5.metric("平均超额", f"{avg_v5_excess:.2f}%")
        
        # v5 累计收益曲线
        st.subheader("📈 v5 累计收益曲线 (选择股票)")
        
        v5_stocks = sorted(backtest_results_v5['symbol'].unique())
        selected_v5_stock = st.selectbox("选择股票", v5_stocks, key='v5_stock')
        
        if not backtest_daily_v5.empty:
            daily_v5 = backtest_daily_v5[backtest_daily_v5['symbol'] == selected_v5_stock].sort_values('trade_date')
            
            if not daily_v5.empty:
                fig_cum_v5 = go.Figure()
                
                fig_cum_v5.add_trace(go.Scatter(
                    x=daily_v5['trade_date'],
                    y=daily_v5['nav'],
                    mode='lines',
                    name='v5策略',
                    line=dict(color='#11998e', width=2.5)
                ))
                
                # 基准线
                first_close = daily_v5['close'].iloc[0]
                benchmark_nav = daily_v5['close'] / first_close
                fig_cum_v5.add_trace(go.Scatter(
                    x=daily_v5['trade_date'],
                    y=benchmark_nav,
                    mode='lines',
                    name='买入持有',
                    line=dict(color='#95a5a6', width=1.5, dash='dash')
                ))
                
                fig_cum_v5.update_layout(
                    title=f"{selected_v5_stock} v5 策略净值 (2024-2025)",
                    xaxis_title="日期",
                    yaxis_title="净值",
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_cum_v5, use_container_width=True)
                
                # 仓位变化
                st.subheader("📊 仓位变化与预警状态")
                
                fig_pos_v5 = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                            row_heights=[0.6, 0.4], vertical_spacing=0.08)
                
                fig_pos_v5.add_trace(go.Scatter(
                    x=daily_v5['trade_date'],
                    y=daily_v5['position'] * 100,
                    mode='lines',
                    name='实际仓位(%)',
                    line=dict(color='#3498db', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(52, 152, 219, 0.2)'
                ), row=1, col=1)
                
                colors_alert = ['#2ecc71' if a == 0 else '#f39c12' if a == 1 else '#e67e22' if a == 2 else '#e74c3c' 
                               for a in daily_v5['alerts']]
                fig_pos_v5.add_trace(go.Bar(
                    x=daily_v5['trade_date'],
                    y=daily_v5['alerts'],
                    name='预警指标数',
                    marker_color=colors_alert,
                    opacity=0.7
                ), row=2, col=1)
                
                fig_pos_v5.update_layout(
                    title=f"{selected_v5_stock} 仓位管理与预警",
                    height=600,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig_pos_v5.update_yaxes(title_text="仓位(%)", row=1, col=1)
                fig_pos_v5.update_yaxes(title_text="预警指标数", row=2, col=1)
                
                st.plotly_chart(fig_pos_v5, use_container_width=True)
            else:
                st.info(f"暂无 {selected_v5_stock} 的v5每日回测数据。")
        else:
            st.warning("backtest_daily_v5 表数据为空。")
    else:
        st.info("v5 回测数据尚未生成。运行 `python backtest_v5.py` 生成。")
    
    # 关键洞察
    st.markdown("---")
    st.subheader("💡 关键洞察")
    st.markdown("""
    <div class="metric-card">
        <p><b>训练集扩展:</b> 从4只大盘股扩展到 <b>14只A股大盘股</b>（新增茅台、宁德时代、平安、招行等），训练数据量从2,340条增至8,190条。</p>
        <p><b>新闻情绪融入:</b> 从东方财富API抓取 <b>614条历史新闻</b>（覆盖42天），提取7个情绪特征（sentiment_1d/3d/7d, major_events, news_count）融入模型。</p>
        <p><b>重大修复:</b> 回测引擎修复 signal_direction=-1 强制空仓的bug，策略收益从接近 0% 跃升至平均 <b>+20.7%</b>。</p>
        <p>三情景调优策略 (adverse=50%, base=80%, min=50%) 在牛市中保留足够敞口，159382.SZ 收益 <b>+71.95%</b> | 588200.SH 收益 <b>+41.65%</b> | 562500.SH 收益 <b>+9.12%</b>。</p>
        <p>加入 Kelly 仓位管理后，最大回撤进一步降低 (159382.SZ 从 -7.7% 降至 -3.8%)，夏普比率提升至 <b>2.09</b>。</p>
        <p><b>模型优化 Round 2:</b> 保留全部60特征 + 手动随机过采样 + 增加树数量(RF 500, LGBM 1000+早停)。Fusion准确率 <b>41.6% → 46.3%</b>，LGBM Binary <b>51.5% → 56.6%</b>，Walk-Forward CV <b>47.94%</b> (+/- 4.80%)。</p>
        <p><b>准确率 vs 收益权衡:</b> Round 2模型更保守（favorable recall仅3%），牛市中仓位偏低，三情景平均收益 <b>+3.1%</b>（vs Round 1 +20.7%）。说明准确率提升≠收益提升，需调整决策阈值使模型在牛市中更激进。</p>
        <p><b>Round 3 趋势跟踪+股灾预判:</b> 引入动态加仓机制（120日收益>15%加仓20%）和多指标股灾预警（波动率/回撤/流动性/暴跌）。平均收益 <b>+10.54%</b>，159382.SZ 收益 <b>+27.85%</b>（超额+3.72%），588200.SH 收益 <b>+29.22%</b>。预警系统在牛市中误报较多，需进一步优化阈值。</p>
        <p><b>v5 20年长周期模型:</b> 训练集扩展至 <b>22只大盘股 × 20年 = 107,248条</b>（vs v4的8,190条，13倍提升）。覆盖2007牛市/2008金融危机/2015杠杆牛/2018贸易战/2020疫情等完整周期。Fusion准确率 <b>56.74%</b>（vs v4 46.3%）。2024-2025测试集平均收益 <b>+0.51%</b>，在A股分化市中保持正收益，胜率~50%。</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== 页脚 ====================
st.sidebar.markdown("---")
st.sidebar.markdown("<div style='text-align: center; color: #888;'>QuantTrade v2.0</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='text-align: center; color: #888;'>© 2026 QuantTrade Team</div>", unsafe_allow_html=True)
