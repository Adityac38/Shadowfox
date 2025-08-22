import pandas as pd
import plotly.express as px
import gradio as gr
def load_data():
    df = pd.read_csv("superstore.csv", encoding='ISO-8859-1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df.dropna(inplace=True)
    return df
df = load_data()
def plot_sales_trend():
    monthly_sales = df.groupby(df['Order Date'].dt.to_period('M')).sum(numeric_only=True).reset_index()
    monthly_sales['Order Date'] = monthly_sales['Order Date'].dt.to_timestamp()
    fig = px.line(monthly_sales, x='Order Date', y='Sales', title='Monthly Sales Trend')
    return fig
def category_analysis():
    category_sales = df.groupby(['Category', 'Sub-Category']).sum(numeric_only=True).reset_index()
    fig = px.bar(category_sales, x='Sub-Category', y='Sales', color='Category', title='Category vs Sub-Category Sales')
    return fig
def product_profit_analysis():
    product_profit = df.groupby('Product Name').sum(numeric_only=True).sort_values(by='Profit', ascending=False).head(10).reset_index()
    fig = px.bar(product_profit, x='Product Name', y='Profit', title='Top 10 Profitable Products')
    return fig
def segment_profit_analysis():
    segment_profit = df.groupby('Segment').sum(numeric_only=True).reset_index()
    fig = px.pie(segment_profit, names='Segment', values='Profit', title='Profit by Customer Segment')
    return fig
def sales_profit_ratio():
    df['Sales to Profit Ratio'] = df['Profit'] / df['Sales']
    ratio_df = df.groupby('Category').mean(numeric_only=True).reset_index()
    fig = px.bar(ratio_df, x='Category', y='Sales to Profit Ratio', title='Sales to Profit Ratio by Category')
    return fig
with gr.Blocks() as demo:
    gr.Markdown("# 🛍️ Store Sales and Profit Analysis Dashboard")
    gr.Markdown("Analyze your retail data interactively with Python + Gradio + Plotly")
    with gr.Tabs():
        with gr.TabItem("📈 Sales Trend"):
            gr.Plot(plot_sales_trend)
        with gr.TabItem("📊 Category Analysis"):
            gr.Plot(category_analysis)
        with gr.TabItem("💰 Product Profit"):
            gr.Plot(product_profit_analysis)
        with gr.TabItem("👥 Segment Profit"):
            gr.Plot(segment_profit_analysis)
        with gr.TabItem("📉 Sales-Profit Ratio"):
            gr.Plot(sales_profit_ratio)
demo.launch()