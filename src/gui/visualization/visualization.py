import plotly.graph_objects as go
import matplotlib.pyplot as plt

def draw_bar_chart(percentages):
    categories = ['Tiêu cực', 'Trung tính', 'Tích cực']
    values = [percentage * 100 for percentage in percentages]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(categories, values, color='b', alpha=0.7)
    
    plt.title('Polarity')
    plt.xlabel('Sentiment')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)
    
    for i, value in enumerate(values):
        ax.text(i, value + 1, f"{value:.2f}%", ha='center', va='bottom')
    
    return fig

def draw_pie_chart_document(percentages):
    categories = ['Tiêu cực', 'Trung tính', 'Tích cực']
    values = [percentage * 100 for percentage in percentages]

    fig = go.Figure(data=[go.Pie(labels=categories, values=values,
                                 textinfo='label+percent',
                                 marker=dict(colors=['#ff9999', '#66b3ff', '#99ff99']),
                                 hole=0)])  # No hole for a standard pie chart

    fig.update_layout(title_text='Sentiment Polarity')
    
    return fig


def draw_pie_chart_paragraph(percentages, labels):
    fig = go.Figure(data=[go.Pie(labels=labels, values=percentages,
                                 textinfo='label+percent',
                                 marker=dict(colors=['#ff9999', '#66b3ff', '#99ff99']))])

    fig.update_layout(title_text='Sentiment Distribution')
    
    return fig