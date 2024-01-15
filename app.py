# Samarpit Vatry
'''
Visualization 1: Hiring Duration Analysis by Career Level
Visualization 2: Hiring Distribution by Career Level
Visualization 3: Hiring Distribution by Business Unit (Exploded Pie chart)
Visualization 4: Hiring Timeline by Business Unit --commented
Visualization 5: Recruited candidates profile --commented
Visualization 6: Number of Requisitions per Hiring Manager by Career Level --commented
Visualization 7: Number of Requisitions by Business Unit
Visualization 8: People Joined by Business Unit
Visualization 9: Tranche vs Age Buckets
Visualization 10: Hiring Status Across Business Units(Data Table) - Function (process_dataframe)
Visualization 11: Hiring Timeline Plotly Graph
'''

# Import Required Libraries
import matplotlib

matplotlib.use('Agg')  # Use Agg backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import io
from base64 import b64encode
from flask import Flask, request, send_file, jsonify, render_template
import numpy as np
import datetime
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import base64
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import plotly.express as px
from plotly.offline import plot
from flask import send_from_directory



LAST_RUN_DF_PATH = "last_dataframe.pkl"
LAST_RUN_TIMESTAMP_PATH = "last_run.txt"


def save_to_pdf(dataframe, output_stream):
    pdf_pages = PdfPages(output_stream, keep_empty=False)  # Use the provided output stream
    images = enhanced_hiring_visualizations(dataframe)

    for img_base64 in images:
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_bytes))

        # Convert image to an array
        img_arr = np.asarray(img)

        # Create a figure and axis to plot the image
        fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100),
                               dpi=100)  # 100 is a conversion factor to get a decent size
        ax.imshow(img_arr)
        ax.axis('off')  # Hide the axes

        # Save the figure to the PDF
        pdf_pages.savefig(fig)

    pdf_pages.close()


# Enhanced Graph Generation Function
def enhanced_hiring_visualizations(dataframe):
    images = []

    # Set global styles
    plt.rcParams['font.family'] = 'Arial'  # or 'Helvetica', 'Arial'
    plt.rcParams['font.size'] = 14

    # Visualization 1: Hiring Duration Analysis by Career Level
    # Create a separate dataframe for Visualization 1
    df1 = dataframe.copy()

    # Convert the relevant columns to datetime format
    df1['Date when shortlist was confirmed'] = pd.to_datetime(df1['Date when shortlist was confirmed'])
    df1['Offered date'] = pd.to_datetime(df1['Offered date'])
    df1['DOJ'] = pd.to_datetime(df1['DOJ'])

    # Calculate the durations for each hiring stage
    df1['REQ_TO_SHORTLIST'] = (df1['Date when shortlist was confirmed'] - df1['Date Created DDMMYYYY']).dt.days
    df1['SHORTLIST_TO_OFFER'] = (df1['Offered date'] - df1['Date when shortlist was confirmed']).dt.days
    df1['OFFER_TO_JOINING'] = (df1['DOJ'] - df1['Offered date']).dt.days

    pivot_analysis_career_level = df1.pivot_table(
        index=['Career Level'],
        values=['REQ_TO_SHORTLIST', 'SHORTLIST_TO_OFFER', 'OFFER_TO_JOINING'],
        aggfunc={'REQ_TO_SHORTLIST': 'mean', 'SHORTLIST_TO_OFFER': 'mean', 'OFFER_TO_JOINING': 'mean'}
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    # colors = plt.cm.Purples(np.linspace(0, 1, 3))
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, 3))
    pivot_analysis_career_level.plot(kind='bar', ax=ax, stacked=True, color=colors, edgecolor='none')

    # Applying dark theme

    # Setting dark theme and other visual attributes for Visualization 1
    labels = ['Requisition to Shortlisting', 'Shortlisting to Offer', 'Offer to Joining']
    ax.set_facecolor('#303030')  # Setting the background color
    fig.set_facecolor('#303030')  # Setting the background color of the figure
    ax.tick_params(axis='both', colors='#eaeaea')  # Setting tick color
    ax.spines['bottom'].set_color('#eaeaea')  # Setting color of x-axis
    ax.spines['left'].set_color('#eaeaea')  # Setting color of y-axis
    ax.yaxis.label.set_color('#eaeaea')  # Setting y-label color
    ax.xaxis.label.set_color('#eaeaea')  # Setting x-label color
    ax.title.set_color('#eaeaea')  # Setting title color
    ax.set_title("Hiring Duration Analysis by Career Level", fontsize=18)
    ax.set_ylabel("Elapsed Time in Days", labelpad=10, fontsize=16)
    ax.set_xlabel("Career Level", labelpad=10, fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    legend = ax.legend(title="Hiring Stages", labels=labels, fontsize=14, bbox_to_anchor=(1.04, 1), loc="upper left",
                       facecolor='#303030', edgecolor='none')
    plt.setp(legend.get_texts(), color='#eaeaea')  # Setting legend text color
    plt.setp(legend.get_title(), color='#eaeaea')  # Setting legend title color
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Displaying numbers on bars
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        if height > 0:
            ax.text(p.get_x() + width / 2,
                    p.get_y() + height / 2,
                    f"{height:.0f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10)
    ...

    # Compute and display the total on top of each stacked bar
    total_heights = pivot_analysis_career_level.sum(axis=1)
    for i, total in enumerate(total_heights):
        ax.text(i, total + 2,  # Adding a small offset for better visibility
                f"{total:.0f}",
                ha="center",
                va="center",
                color="white",
                fontsize=11)
    ...

    plt.grid(False)

    plt.tight_layout()
    # plt.gcf().canvas.draw()
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    images.append(b64encode(img.getvalue()).decode('utf-8'))
    plt.close()

    # Visualization 2: Hiring Distribution by Career Level
    # Create a separate dataframe for Visualization 2
    df2 = dataframe.copy()

    # Convert the relevant columns to datetime format
    df2['Date when shortlist was confirmed'] = pd.to_datetime(df2['Date when shortlist was confirmed'])
    df2['Offered date'] = pd.to_datetime(df2['Offered date'])
    df2['DOJ'] = pd.to_datetime(df2['DOJ'])


    df2['Status (Open/ On-Hold/Pending Completion/ Joined/ Cancelled)'] = df2[
        'Status (Open/ On-Hold/Pending Completion/ Joined/ Cancelled)'].str.strip()
    # Grouping and pivoting for Visualization 2
    grouped_status = df2.groupby(
        ['Career Level', 'Status (Open/ On-Hold/Pending Completion/ Joined/ Cancelled)']).size().reset_index(
        name='Count')
    pivot_status = grouped_status.pivot(index='Career Level',
                                        columns='Status (Open/ On-Hold/Pending Completion/ Joined/ Cancelled)',
                                        values='Count').fillna(0)

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    # pivot_status.plot(kind='barh', stacked=True, ax=ax2, colormap="viridis_r", edgecolor='none')
    colors = plt.cm.Set1(np.arange(len(pivot_status.columns)))
    # colors = plt.cm.tab20(np.arange(len(pivot_status.columns)))

    pivot_status.plot(kind='barh', stacked=True, ax=ax2, color=colors, edgecolor='none')

    # Modified function to place label inside the bars and omit zeros
    def place_label_inside_bar_clean(patches, ax):
        cumulative_widths = {patch.get_y(): 0 for patch in patches}
        for patch in patches:
            width = patch.get_width()
            y_position = patch.get_y()
            if width > 0:  # Omitting zeros
                cumulative_width = cumulative_widths[y_position]
                ax.text(cumulative_width + (width / 2), y_position + 0.55 * patch.get_height(),
                        f'{width:.0f}', ha='center', va='center', color='black', fontsize=11)
                cumulative_widths[y_position] += width

    # Place labels inside bars
    place_label_inside_bar_clean(ax2.patches, ax2)

    # Settings for the graph appearance
    ax2.set_facecolor('#303030')
    fig2.set_facecolor('#303030')
    ax2.tick_params(axis='both', colors='#eaeaea')
    ax2.spines['bottom'].set_color('#eaeaea')
    ax2.spines['left'].set_color('#eaeaea')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.label.set_color('#eaeaea')
    ax2.xaxis.label.set_color('#eaeaea')
    ax2.title.set_color('#eaeaea')
    ax2.grid(False)

    plt.title('Hiring Distribution by Career Level', fontsize=18)
    plt.xlabel('Number of Candidates', fontsize=16)
    plt.ylabel('Career Level', fontsize=16)
    legend = ax2.legend(title="Hiring Status", fontsize=10, facecolor='none', edgecolor='none',
                        bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.setp(legend.get_texts(), color='#eaeaea')
    plt.setp(legend.get_title(), color='#eaeaea')

    plt.tight_layout()
    # plt.gcf().canvas.draw()
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    images.append(b64encode(img.getvalue()).decode('utf-8'))
    plt.close()

    # Visualization 3: Hiring Distribution by Business Unit (Exploded Pie chart)
    # Create a separate dataframe for Visualization 3
    df3 = dataframe.copy()

    df3["Business Vertical"] = df3["Business Vertical"].str.strip().replace("Fixed networks", "Fixed Networks")

    business_vertical_counts = df3['Business Vertical'].value_counts()
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    explode = [0.1] * len(business_vertical_counts)
    colors = sns.color_palette("deep", n_colors=len(business_vertical_counts))

    # Custom function to display both percentage and count in pie chart labels
    def func(pct, allvals):
        absolute = round(pct / 100. * np.sum(allvals))
        return "{:.1f}%\n({:d} reqs)".format(pct, absolute)

    business_vertical_counts.plot(kind='pie', autopct=lambda p: func(p, business_vertical_counts), startangle=210,
                                  explode=explode, colors=colors, ax=ax3, textprops={'color': "white", 'fontsize': 13},
                                  wedgeprops=dict(edgecolor='none'))

    # Setting the dark theme for Visualization 3
    ax3.set_facecolor('#303030')  # Setting the background color for the axis
    fig3.set_facecolor('#303030')  # Setting the background color of the figure
    ax3.title.set_color('#eaeaea')  # Setting title color

    # Removing ylabel and setting title
    ax3.set_ylabel('')
    plt.title('Hiring Distribution by Business Units', fontsize=18, color='#eaeaea')
    # Removing the legend
    ax3.legend().set_visible(False)

    plt.tight_layout()
    # plt.gcf().canvas.draw()
    img = io.BytesIO()
    plt.savefig(img, format="png", facecolor=fig3.get_facecolor())
    img.seek(0)
    images.append(b64encode(img.getvalue()).decode('utf-8'))
    plt.close()

    # Visualization 4: Hiring Timeline
    # Create a separate dataframe for Visualization 4
    df4 = dataframe.copy()

    # Data preprocessing and transformations
    df4["Business Vertical"] = df4["Business Vertical"].str.strip().replace("Fixed networks", "Fixed Networks")
    df4 = df4.rename(columns={'Business Vertical': 'Unit', 'DOJ': 'Joining', 'Offered date': 'Offered'})
    units = df4['Unit'].dropna().unique()

    # Compute the "Offered" and "Joined" counts for each business unit
    offered_counts = df4.groupby('Unit')['Offered'].count()
    joined_counts = df4.groupby('Unit').apply(
        lambda x: x[['Joined  Male', 'Joined  Female', 'Joined  PWD']].notna().sum(axis=1).sum())

    # Load the custom icons from the 'static' folder
    male_icon = plt.imread("static/man.png")
    female_icon = plt.imread("static/female.png")
    pwd_icon = plt.imread("static/disabled.png")

    # Functions to plot custom icons on the visualization
    def plot_icon(x, y, icon, ax, zoom=0.04):
        im = OffsetImage(icon, zoom=zoom)
        ab = AnnotationBbox(im, (x, y), frameon=False, pad=0.1)
        ax.add_artist(ab)

    def plot_icon_with_check(x, y, icon, ax, zoom=0.04):
        if pd.notna(x):
            plot_icon(x, y, icon, ax, zoom)

    # Visualization code
    vibrant_palette = sns.color_palette("husl", n_colors=len(units))
    fig4, ax4 = plt.subplots(figsize=(18, 10))
    sns.set_style("whitegrid")
    sns.scatterplot(data=df4, x='Offered', y='Unit', s=150, hue='Unit', palette=vibrant_palette, legend=None, ax=ax4)

    # Plot custom icons based on data
    for idx, row in df4.iterrows():
        if not pd.isna(row['Joined  Male']):
            plot_icon_with_check(row['Joining'], row['Unit'], male_icon, ax4)
        if not pd.isna(row['Joined  Female']):
            plot_icon_with_check(row['Joining'], row['Unit'], female_icon, ax4)
        if not pd.isna(row['Joined  PWD']):
            plot_icon_with_check(row['Joining'], row['Unit'], pwd_icon, ax4)

    # Setting dark theme for the visualization
    ax4.set_facecolor('#303030')  # Setting the background color for the axis
    fig4.set_facecolor('#303030')  # Setting the background color of the figure
    ax4.tick_params(axis='both', colors='#eaeaea')  # Setting tick color
    ax4.spines['bottom'].set_color('#eaeaea')  # Setting color of x-axis
    ax4.spines['left'].set_color('#eaeaea')  # Setting color of y-axis
    ax4.yaxis.label.set_color('#eaeaea')  # Setting y-label color
    ax4.xaxis.label.set_color('#eaeaea')  # Setting x-label color
    ax4.title.set_color('#eaeaea')  # Setting title color

    plt.title("Hiring Timeline by Business Unit", fontsize=16, pad=20, color='#eaeaea')
    plt.xlabel("Timeline", fontsize=15, labelpad=15, color='#eaeaea')
    plt.ylabel("Business Unit", fontsize=15, labelpad=15, color='#eaeaea')
    plt.xticks(fontsize=14, color='#eaeaea')
    plt.yticks(fontsize=14, color='#eaeaea')

    # Compute the "Requisition" counts for each business unit
    req_counts = df4.groupby('Unit').size()

    # Display "Requisition", "Offered", and "Joined" counts inside the shaded bars for each business unit
    for idx, unit in enumerate(units):
        ax4.text(df4['Offered'].min(), idx + 0.3, f"No of Reqs: {req_counts[unit]}", color='#eaeaea', va='center',
                 ha='left', fontsize=10, fontweight='bold')
        ax4.text(df4['Offered'].mean(), idx + 0.3, f"Offered: {offered_counts[unit]}", color='#eaeaea', va='center',
                 ha='center', fontsize=10, fontweight='bold')
        ax4.text(df4['Offered'].max(), idx + 0.3, f"Joined: {joined_counts[unit]}", color='#eaeaea', va='center',
                 ha='right', fontsize=10, fontweight='bold')

    for idx, unit in enumerate(units):
        plt.axhspan(idx + 0.5, idx - 0.5, color=vibrant_palette[idx], alpha=0.1)

    ax4.grid(True, which='both', linestyle='--', linewidth=0.5)  # Displaying the grid
    sns.despine(left=True, bottom=True)
    # Set the x-axis limits to 15th March and 15th November 2023
    start_date = np.datetime64('2023-03-15')
    end_date = np.datetime64('2023-11-15')
    ax4.set_xlim(start_date, end_date)

    # Set custom ticks for months between 15th March and 15th November 2023
    months = [np.datetime64(f'2023-{month:02d}-15') for month in range(3, 12)]
    ax4.set_xticks(months)
    ax4.set_xticklabels([f"{month.strftime('%b')}-23" for month in pd.to_datetime(months)])

    plt.tight_layout()
    # plt.gcf().canvas.draw()
    img = io.BytesIO()
    plt.savefig(img, format="png", facecolor=fig4.get_facecolor())
    img.seek(0)
    images.append(b64encode(img.getvalue()).decode('utf-8'))
    plt.close()

    # Visualization 5: Recruited candidates profile
    # Create a separate dataframe for Visualization 5
    df5 = dataframe.copy()

    offered_columns = ['Offered Male', 'Offered Female', 'Offered PWD', 'Offered Others']
    joined_columns = ['Joined  Male', 'Joined  Female', 'Joined  PWD', 'Joined  Others']
    offered_data = df5[offered_columns].sum()
    joined_data = df5[joined_columns].sum()
    categories = ['Male', 'Female', 'PWD', 'Others']
    offered_data.index = categories
    joined_data.index = categories
    combined_data = pd.DataFrame({'Offered': offered_data, 'Joined': joined_data})
    fig, ax = plt.subplots(figsize=(12, 8))
    # colors = ['#FF7F50', '#FFD700']  # Coral & Gold
    colors = ['#EA738D', '#89ABE3']
    bar_width = 0.35
    index = np.arange(len(combined_data))

    bar1 = ax.bar(index, combined_data['Offered'], bar_width, label='Offered', color=colors[0], edgecolor='none',
                  alpha=0.8)
    bar2 = ax.bar([i + bar_width for i in index], combined_data['Joined'], bar_width, label='Joined', color=colors[1],
                  edgecolor='none', alpha=0.8)

    # Function to add counts inside bars without decimals and omitting zeros
    def add_labels_inside_no_decimal(bars):
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:  # Only show non-zero values
                ax.text(bar.get_x() + bar.get_width() / 2, yval / 2, str(int(yval)), ha='center', va='center',
                        fontsize=13, color='black')

    add_labels_inside_no_decimal(bar1)
    add_labels_inside_no_decimal(bar2)

    # Applying dark theme for Visualization 5
    ax.set_facecolor('#303030')
    fig.set_facecolor('#303030')
    ax.tick_params(axis='both', colors='#eaeaea')
    ax.spines['bottom'].set_color('#eaeaea')
    ax.spines['left'].set_color('#eaeaea')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.label.set_color('#eaeaea')
    ax.xaxis.label.set_color('#eaeaea')
    ax.title.set_color('#eaeaea')

    ax.set_title('Recruited candidates profile', fontsize=22)
    ax.set_xlabel('Categories', fontsize=18)
    ax.set_ylabel('Number of Candidates', fontsize=18)
    legend = ax.legend(fontsize=16, facecolor='none', edgecolor='none', prop={'size': 18})
    plt.setp(legend.get_texts(), color='#eaeaea')  # Setting legend text color

    ax.grid(False)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(categories, fontsize=17)
    ax.tick_params(axis='y', labelsize=17)
    plt.tight_layout()
    # plt.gcf().canvas.draw()
    img = io.BytesIO()
    plt.savefig(img, format="png", facecolor=fig.get_facecolor())
    img.seek(0)
    images.append(b64encode(img.getvalue()).decode('utf-8'))
    plt.close()

    # Visualization 6: Number of Requisitions per Hiring Manager by Career Level
    # Create a separate dataframe for Visualization 6
    df6 = dataframe.copy()

    grouped_data = df6.groupby(['Hiring Manager Name', 'Career Level']).size().reset_index(name='Count')
    pivot_data = grouped_data.pivot(index='Hiring Manager Name', columns='Career Level', values='Count').fillna(0)
    color_palette = plt.cm.viridis(np.linspace(0, 1, len(pivot_data.columns)))
    fig, ax = plt.subplots(figsize=(15, 10))
    pivot_data.plot(kind='barh', stacked=True, ax=ax, color=color_palette, edgecolor='none')  # No edgecolor for bars

    # Applying dark theme for Visualization 6
    ax.set_facecolor('#303030')
    fig.set_facecolor('#303030')
    ax.tick_params(axis='both', colors='#eaeaea')
    ax.spines['bottom'].set_color('#eaeaea')
    ax.spines['left'].set_color('#eaeaea')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.label.set_color('#eaeaea')
    ax.xaxis.label.set_color('#eaeaea')
    ax.title.set_color('#eaeaea')

    plt.title('Number of Requisitions per Hiring Manager', fontsize=14)  # Reduced font size for title
    plt.ylabel('Hiring Manager Name', fontsize=13)  # Reduced font size for Y-axis label
    plt.xlabel('Count of Requisitions', fontsize=13)  # Reduced font size for X-axis label
    plt.yticks(fontsize=10)  # Reduced font size for Y-axis ticks

    legend = ax.legend(title='Career Level', fontsize=12, facecolor='none', edgecolor='none', bbox_to_anchor=(1.04, 1),
                       loc="upper left")
    plt.setp(legend.get_texts(), color='#eaeaea')  # Setting legend text color
    plt.setp(legend.get_title(), color='#eaeaea', fontsize=12)  # Setting legend title color and reduced font size

    plt.gca().yaxis.grid(False)
    plt.tight_layout()
    # plt.gcf().canvas.draw()
    img = io.BytesIO()
    plt.savefig(img, format="png", facecolor=fig.get_facecolor())
    img.seek(0)
    images.append(b64encode(img.getvalue()).decode('utf-8'))
    plt.close()

    # Visualization 7: Number of Requisitions by Business Unit
    df7 = dataframe.copy()

    df7["Business Vertical"] = df7["Business Vertical"].str.strip().replace("Fixed networks", "Fixed Networks")

    # Grouping and pivoting data by 'Business Vertical' and 'Career Level'
    grouped_data_vertical = df7.groupby(['Business Vertical', 'Career Level']).size().reset_index(name='Count')
    pivot_data_vertical = grouped_data_vertical.pivot(index='Business Vertical', columns='Career Level',
                                                      values='Count').fillna(0)

    fig, ax = plt.subplots(figsize=(15, 10))
    # color_palette = plt.cm.cividis(np.linspace(0.2, 1, len(pivot_data_vertical.columns)))
    color_palette = plt.cm.GnBu(np.linspace(0.2, 1, len(pivot_data_vertical.columns)))

    pivot_data_vertical.plot(kind='barh', stacked=True, ax=ax, color=color_palette, edgecolor='none')
    for p in ax.patches:
        left, bottom, width, height = p.get_bbox().bounds
        if width > 0:
            ax.text(left + width / 2, bottom + height / 2, int(width), ha='center', va='center', color='black',
                    fontsize=16)

    # Applying dark theme
    ax.set_facecolor('#303030')
    fig.set_facecolor('#303030')
    ax.tick_params(axis='both', colors='#eaeaea')
    ax.tick_params(axis='y', labelsize=19)
    ax.tick_params(axis='x', labelsize=19)
    ax.spines['bottom'].set_color('#eaeaea')
    ax.spines['left'].set_color('#eaeaea')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.label.set_color('#eaeaea')
    ax.xaxis.label.set_color('#eaeaea')
    ax.title.set_color('#eaeaea')

    ax.set_title(f'Number of Requisitions by Business Unit', fontsize=26)
    ax.set_ylabel('Business Vertical', fontsize=24)
    ax.set_xlabel('Count of Requisitions', fontsize=23)

    legend = ax.legend(title='Career Level', fontsize=18, facecolor='none', edgecolor='none', bbox_to_anchor=(1.04, 1),
                       loc="upper left")
    plt.setp(legend.get_texts(), color='#eaeaea')
    plt.setp(legend.get_title(), color='#eaeaea', fontsize=22)

    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    plt.tight_layout()
    # plt.gcf().canvas.draw()
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    images.append(b64encode(img.getvalue()).decode('utf-8'))
    plt.close()

    # Visualization 8: People Joined by Business Unit
    # Create a copy of the dataframe for this visualization
    df8 = dataframe.copy()

    df8["Business Vertical"] = df8["Business Vertical"].str.strip().replace("Fixed networks", "Fixed Networks")
    # Replacing NaN values with zeros in the "Joined" columns
    df8[['Joined  Male', 'Joined  Female', 'Joined  PWD', 'Joined  Others']] = df8[
        ['Joined  Male', 'Joined  Female', 'Joined  PWD', 'Joined  Others']].fillna(0)

    # Grouping by 'Business Vertical'
    grouped_data_vertical = df8.groupby("Business Vertical")[
        ['Joined  Male', 'Joined  Female', 'Joined  PWD', 'Joined  Others']].sum()
    # Renaming the columns of the DataFrame before transposing
    grouped_data_vertical.columns = ['Male', 'Female', 'PWD', 'Others']

    # Setting the figure background and axes background to dark grey
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#303030')  # Dark grey figure background
    ax.set_facecolor('#303030')  # Dark grey axes background

    # Adjusting text color for legibility against dark background
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(axis='both', colors='white')

    # Heatmap visualization with a colormap suitable for a dark theme #YlGnBu #coolwarm #RdGy #Greys #GnBu #YlOrBr
    sns.heatmap(grouped_data_vertical.T, cmap="YlOrBr", annot=True, fmt="g",
                cbar_kws={'label': 'Number of People Joined'}, linewidths=0.5, linecolor='#303030')
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    plt.title("People Joined by Business Unit", fontsize=18)
    plt.tight_layout()
    # plt.gcf().canvas.draw()
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    images.append(b64encode(img.getvalue()).decode('utf-8'))
    plt.close()

    # Visualization 9: Tranche vs Age Buckets
    df9 = dataframe.copy()  # Create a copy of dataframe for this visualization
    # Filter out rows with NaN in either of the two columns
    df_clean = df9.dropna(subset=["Tranche 1A/1B/2", "Age Bucket"])
    # Correcting the Age Bucket range inconsistency
    df_clean["Age Bucket"] = df_clean["Age Bucket"].str.replace(' ', '')
    # List of specified Age Bucket values
    age_buckets_specified = ['0-30', '31-60', '61-90', '91-120', '121-150', '151-180']
    # List of specified Tranche values
    tranche_values_specified = ['1A', '1B', 2]
    # Filter the dataframe for the specified values
    df_filtered = df_clean[
        df_clean["Age Bucket"].isin(age_buckets_specified) & df_clean["Tranche 1A/1B/2"].isin(tranche_values_specified)]

    # Create a pivot table for plotting
    pivot_df_filtered = df_filtered.groupby(['Age Bucket', 'Tranche 1A/1B/2']).size().unstack().fillna(0)
    # Reorder the pivot table based on the sorted age buckets
    pivot_df_filtered = pivot_df_filtered.reindex(age_buckets_specified)

    # Set the dark grey background for the graph
    fig, ax = plt.subplots(figsize=(10, 7))
    # Applying dark theme
    ax.set_facecolor('#303030')
    fig.set_facecolor('#303030')
    ax.tick_params(axis='both', colors='#eaeaea')
    ax.spines['bottom'].set_color('#eaeaea')
    ax.spines['left'].set_color('#eaeaea')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.label.set_color('#eaeaea')
    ax.xaxis.label.set_color('#eaeaea')
    ax.title.set_color('#eaeaea')

    pivot_df_filtered.plot(kind='barh', stacked=True, colormap="Pastel1", ax=ax, edgecolor='none')
    # Set the title and adjust its color
    ax.set_title('Tranche vs Age Buckets', color='white')
    # Set labels and their colors for x and y axes
    ax.set_xlabel('Count', color='white')
    ax.set_ylabel('Age Bucket', color='white')

    # Adjust tick colors for x and y axes
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.tight_layout()
    # plt.gcf().canvas.draw()
    ax.grid(False)  # Remove grid lines
    cumulative_widths = [0] * len(pivot_df_filtered)

    # Add count annotations to the bars with adjusted positions (excluding zeros)
    for i, rect in enumerate(ax.patches):
        width = rect.get_width()
        if width > 0:  # Check to ensure width is not zero
            y_pos = rect.get_y() + rect.get_height() / 2
            ax.text(cumulative_widths[i % len(cumulative_widths)] + width / 2, y_pos,
                    int(width), ha='center', va='center', color='black', fontsize=10)
        cumulative_widths[i % len(cumulative_widths)] += width

    legend = ax.legend(title="Tranche", loc="upper right", facecolor='#303030', edgecolor='none')
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')
    # Save the plot to a BytesIO object and then append it to the images list
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    images.append(b64encode(img.read()).decode('utf-8'))
    plt.close(fig)

    return images


# Visualization 10: Hiring Status Across Business Units (Data Table)
def process_dataframe(df):
    df10 = df.copy()
    column_mapping = {
        'Business Vertical': 'Unit',
        'Hiring Manager Name': 'Hiring Manager',
        'LT-1': 'Senior Manager',
        'Tranche 1A/1B/2': 'Tranche',
        'Status (Open/ On-Hold/Pending Completion/ Joined/ Cancelled)': 'Status'
    }
    df10.rename(columns=column_mapping, inplace=True)
    #df10 = df10[list(column_mapping.values())].fillna('Undefined')
    df10 = df10[list(column_mapping.values())].apply(
        lambda col: col.fillna('N/A') if col.name == 'Tranche' else col.fillna('Undefined'))

    #df10.loc[df10['Unit'].str.strip().str.lower() == 'fixed network', 'Tranche'] = 'N/A'
    df10['Unit'] = df10['Unit'].str.strip().apply(
        lambda x: 'Fixed Network' if x.lower() == 'fixed network' else ('NAS' if x.lower() == 'nas' else x.title()))

    df10['Status'] = df10['Status'].str.strip().str.title()
    df10['Status'].fillna('Undefined', inplace=True)


    grouping_columns = ['Unit', 'Senior Manager', 'Hiring Manager', 'Tranche']
    grouped_df = df10.groupby(grouping_columns + ['Status'])['Status'].size().reset_index(name='Total Roles')
    final_pivot_df = grouped_df.pivot_table(index=grouping_columns, columns='Status', values='Total Roles',
                                            aggfunc='sum', fill_value=0).reset_index()
    final_pivot_df['Total Roles'] = final_pivot_df.iloc[:, -len(df10['Status'].unique()):].sum(axis=1)

    return final_pivot_df


# Visualization 11: Hiring Timeline Plotly Graph
def generate_plotly_graph(dataframe):
    df4 = dataframe.copy()
    # Convert the relevant columns to datetime format
    df4['Date when shortlist was confirmed'] = pd.to_datetime(df4['Date when shortlist was confirmed'])
    df4['Offered date'] = pd.to_datetime(df4['Offered date'])
    df4['DOJ'] = pd.to_datetime(df4['DOJ'])
    df4["Business Vertical"] = df4["Business Vertical"].str.strip().replace("Fixed networks", "Fixed Networks")
    df4 = df4.rename(columns={'Business Vertical': 'Unit', 'DOJ': 'Joining', 'Offered date': 'Offered'})

    # Replace "Unknown" with a default date to represent missing data
    default_date = pd.Timestamp('1900-01-01')
    df4['Offered'].fillna(default_date, inplace=True)
    df4['Unit'].fillna("Unknown", inplace=True)

    # Aggregating the data for each date and unit combination to get the list of candidates who joined
    df4['Candidate Name'] = df4['Candidate Name'].astype(str)
    grouped = df4.groupby(['Joining', 'Unit']).agg(
        {'Candidate Name': ', '.join, 'Joined  Male': 'sum', 'Joined  Female': 'sum',
         'Joined  PWD': 'sum'}).reset_index()

    # Filter out rows where 'Unit' is "Unknown" or blank
    grouped_filtered = grouped[grouped['Unit'] != "Unknown"]

    # Create an empty scatter plot
    fig = px.scatter(title="Hiring Timeline by Business Unit",
                     labels={'Joining': 'Timeline'},
                     color_discrete_sequence=px.colors.qualitative.Set1)

    # Unicode symbols for categories
    symbols = {
        'Male': '🤵',
        'Female': '👩🏻‍💼',
        'PWD': '🦽',
        'Others': '🤵'
    }
    # ♿ 🤵

    # Add markers with the adjusted hover text format for Male, Female, and PWD
    for _, row in grouped_filtered.iterrows():
        names = ', '.join(row['Candidate Name'].split(', '))

        # Determine marker symbol and color based on joined candidates
        if row['Joined  Male'] > 0:
            marker_color = 'gold'
            marker_symbol = symbols['Male']
        elif row['Joined  Female'] > 0:
            marker_color = 'coral'
            marker_symbol = symbols['Female']
        elif row['Joined  PWD'] > 0:
            marker_color = 'purple'
            marker_symbol = symbols['PWD']
        else:
            marker_color = 'teal'
            marker_symbol = symbols['Others']

        fig.add_scatter(x=[row['Joining']], y=[row['Unit']], mode='markers+text',
                        text=marker_symbol, textposition='middle center',
                        marker_color=marker_color, marker_size=30, marker_opacity=0.01,
                        hovertext=f"{names}<br>{row['Joining'].strftime('%d-%b-%Y')}", hoverinfo='text', textfont_size=25)

    # Applying the layout
    units_filtered = grouped_filtered['Unit'].dropna().unique()
    unit_to_numeric_filtered = {unit: idx for idx, unit in enumerate(units_filtered)}

    fig.update_layout(
        title=dict(text="Hiring Timeline by Business Unit", x=0.5, xanchor='center',
                   font=dict(size=20, family="Arial")),
        xaxis=dict(title="Timeline", showgrid=True, gridcolor="#999999", zeroline=False,
                   title_font=dict(size=18, family="Arial")),
        yaxis=dict(title="Business Unit", showgrid=True, gridcolor="#999999", zeroline=False,
                   title_font=dict(size=18, family="Arial")),
        showlegend=False,
        plot_bgcolor="#303030",
        paper_bgcolor="#303030",
        font=dict(family="Arial", color="#eaeaea"),
        bargap=0,
    )

    # Set x-axis limits
    start_date = '2023-04-15'
    end_date = '2024-01-15'

    # Convert start_date and end_date to datetime objects
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    # Modify the part of the code that adds the colored background rectangles to span the entire timeline
    for idx, unit in enumerate(units_filtered):
        y_numeric = unit_to_numeric_filtered[unit]
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=start_date_dt,
            x1=end_date_dt,
            y0=y_numeric - 0.5,
            y1=y_numeric + 0.5,
            fillcolor=px.colors.qualitative.Set1[idx],
            opacity=0.1,
            layer="below",
            line_width=0,
        )

    # Compute the "Requisition" counts for each business unit
    req_counts = df4.groupby('Unit').size()

    # Compute the "Offered" and "Joined" counts for each business unit
    offered_rows = df4[(df4['Offered Male'] == 1) |
                       (df4['Offered Female'] == 1) |
                       (df4['Offered PWD'] == 1) |
                       (df4['Offered Others'] == 1)]
    offered_counts = offered_rows.groupby('Unit').size()
    joined_counts = df4.groupby('Unit').apply(
        lambda x: x[['Joined  Male', 'Joined  Female', 'Joined  PWD']].notna().sum(axis=1).sum())

    # Display "Requisition", "Offered", and "Joined" counts on the scatter plot for each business unit
    for idx, unit in enumerate(units_filtered):
        y_numeric = unit_to_numeric_filtered[unit]
        fig.add_annotation(
            x=start_date_dt + pd.DateOffset(months=1),
            y=y_numeric,
            text=f"No of Reqs: {req_counts[unit]}",
            showarrow=False,
            yshift=20,
            font=dict(color="#eaeaea", size=10, family="Arial")
        )
        fig.add_annotation(
            x=pd.Timestamp((start_date_dt.value + end_date_dt.value) // 2),
            y=y_numeric,
            text=f"Offered: {offered_counts.get(unit, 0)}",
            showarrow=False,
            yshift=20,
            font=dict(color="#eaeaea", size=10, family="Arial")
        )
        fig.add_annotation(
            x=end_date_dt - pd.DateOffset(months=1),
            y=y_numeric,
            text=f"Joined: {joined_counts[unit]}",
            showarrow=False,
            yshift=20,
            font=dict(color="#eaeaea", size=10, family="Arial")
        )

    fig.update_xaxes(range=[start_date, end_date])

    # Save the figure to an HTML file and return its name
    filename = "plotly_timeline_graph.html"
    plot(fig, filename=filename)
    return filename
    #return plot(fig, output_type='div', include_plotlyjs=False)


# Flask Application with Enhanced UI
app = Flask(__name__)

PLOTLY_FIG_DATA_PATH = "plotly_fig_data.json"


@app.route('/', methods=['GET', 'POST'])
def index():
    message = None
    images = []
    error = None
    datatable_html = ""
    units = ["All"]
    plot_div = ""

    if request.method == 'POST':
        file = request.files.get('file')
        if file.filename == '':
            error = "Please upload a file."
        else:
            if not file.filename.endswith(('.xls', '.xlsx')):
                error = "Please upload a valid Excel file."
            else:
                try:
                    df = pd.read_excel(file, header=2)
                    df.columns = df.columns.str.strip()
                    df.to_pickle(LAST_RUN_DF_PATH)
                    with open(LAST_RUN_TIMESTAMP_PATH, "w") as f:
                        f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                    processed_df = process_dataframe(df)  # Processing the DataFrame to get final_pivot_df
                    if processed_df is not None:  # Check if processed_df is not None before accessing it
                        units.extend([unit for unit in processed_df['Unit'].unique().tolist() if unit != "Undefined"])


                except Exception as e:
                    error = f"Error reading the Excel file: {str(e)}"

        if not error:
            try:
                images = enhanced_hiring_visualizations(df)
                filename = generate_plotly_graph(df)
                plot_div = f'<iframe src="/{filename}" width="100%" height="600"></iframe>'

                if images:
                    datatable_html = base64.b64decode(images[-1]).decode('utf-8')
                    images = images[:-1]
                message = "Visualizations generated successfully!"
            except Exception as e:
                error = f"Error generating visualizations: {str(e)}"

    elif os.path.exists(LAST_RUN_DF_PATH):
        df = pd.read_pickle(LAST_RUN_DF_PATH)
        processed_df = process_dataframe(df)
        units = [unit for unit in processed_df['Unit'].unique().tolist() if unit != "Undefined"]

        last_run_time = None
        if os.path.exists(LAST_RUN_TIMESTAMP_PATH):
            with open(LAST_RUN_TIMESTAMP_PATH, "r") as f:
                last_run_time = f.read().strip()

        if os.path.exists("plotly_graph.html"):
            plot_div = '<iframe src="/plotly_graph" width="100%" height="600"></iframe>'

        try:
            images = enhanced_hiring_visualizations(df)

            if last_run_time:
                message = f"Visualizations generated from last upload on {last_run_time}"
            else:
                message = "Visualizations generated from last upload"
        except Exception as e:
            error = f"Error generating visualizations: {str(e)}"

    return render_template('index.html', images=images, error=error, message=message, datatable_html=datatable_html,
                           units=units, plot_div=plot_div)



# Handle internal server errors to provide user-friendly feedback
@app.errorhandler(500)
def internal_server_error(e):
    error = "An internal error occurred. Please try again later."
    return render_template('index.html', error=error)

@app.route('/plotly_graph', methods=['GET'])
def serve_plotly_graph():
    return send_from_directory('.', 'plotly_timeline_graph.html')


@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    if os.path.exists(LAST_RUN_DF_PATH):  # Check if the DataFrame pickle file exists
        df_last_run = pd.read_pickle(LAST_RUN_DF_PATH)  # Load the DataFrame

        # Use a BytesIO object to capture the PDF data
        pdf_stream = io.BytesIO()
        save_to_pdf(df_last_run, pdf_stream)  # Modify the save_to_pdf function to accept an output stream

        # Set the stream position to the beginning
        pdf_stream.seek(0)

        # Directly send the stream to the user
        return send_file(pdf_stream, attachment_filename="Hiring_Graphs.pdf", as_attachment=True,
                         mimetype='application/pdf')

    else:
        return "Data not found!", 404


@app.route('/get_data/all', methods=['GET'])
def get_data_all():
    if os.path.exists(LAST_RUN_DF_PATH):
        df = pd.read_pickle(LAST_RUN_DF_PATH)
        processed_df = process_dataframe(df)
        data = processed_df.to_dict(orient='records')
        return jsonify(data)
    else:
        return jsonify([])  # Return an empty list if there is no data available


@app.route('/get_data/<string:unit>', methods=['GET'])
def get_data(unit):
    if os.path.exists(LAST_RUN_DF_PATH):
        df = pd.read_pickle(LAST_RUN_DF_PATH)
        final_pivot_df = process_dataframe(df)  # Processing the DataFrame to get final_pivot_df

        if unit.lower() == 'all':  # Comparing in lowercase to make it case-insensitive
            filtered_df = final_pivot_df
        else:
            filtered_df = final_pivot_df[final_pivot_df['Unit'] == unit]  # Filter by 'Unit'

        # Converting the filtered DataFrame to dictionary format and returning as JSON
        data = filtered_df.to_dict(orient='records')
        return jsonify(data)
    else:
        return jsonify([])  # Return an empty list if there is no data available


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5004, debug=True)