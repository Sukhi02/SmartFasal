from django.shortcuts import render
from django.http import HttpResponse
from .models import FTP_session_model
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['interactive'] = False


def plot_graph(list_1, list_2,label):
    plt.plot(list_1, list_2, label=label)
    plt.ylabel(label)
    plt.xlabel('Time')
    plt.title(label + 'of the Day ')
    plt.legend()
    plt.savefig('static/' + label + '.png')


class FTP_session_view:

    def index(request):
        return render(request,'home.html')

    def visual(request):
        return render(request,'visualise.html')
######################################################################################################
    def jslearning(request):

        ######### Phase 1:- Data aloocating and defining
        import plotly.graph_objects as go
        import numpy as np

        title = 'Readings of the SmartFasal'
        labels = [' S M at 15 cms', 'S M at 40 cms', 'S M at 80 cms', 'Temperature', 'Humidity', 'Pressure', 'Luminisity']
        colors = ['rgb(67,67,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)', 'rgb(67,67,67)', 'rgb(67,67,67)', 'rgb(67,67,67)']

        mode_size = [8, 8, 8, 8, 8, 8, 8]
        line_size = [2, 2, 2, 2, 2, 2, 2]

        x_data = np.vstack((np.arange(0, 23 ),)*7)
        ########### data allocation phase completed.

        ########### phase :: Accessing the downloaded file (Data1.csv)
        import os
        from smartfasal_project.settings import BASE_DIR
        STATIC_ROOT = os.path.join(BASE_DIR, 'static')
        path = STATIC_ROOT
        os.chdir(path)
        import pandas as pd
        filename = "data1.csv"
        Local_data_smartfasal = pd.read_csv(filename, names=['Timestamp', 'S_M_10cm','S_M_45cm','S_M_80cm', 'Temperature', 'Humidity', 'Pressure', 'Luxes'])

        y_data = np.array([
                          Local_data_smartfasal['S_M_10cm'],
                          Local_data_smartfasal['S_M_45cm'],
                          Local_data_smartfasal['S_M_80cm'],
                          Local_data_smartfasal['Temperature'],
                          Local_data_smartfasal['Humidity'],
                          Local_data_smartfasal['Pressure'],
                          Local_data_smartfasal['Luxes'],
                          ])

        ################ Data loading Phase completed

        ############### Phase 3:- Data plotting
        fig = go.Figure()

        for i in range(0, 7):
            fig.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',
                name=labels[i],
                line=dict(color=colors[i], width=line_size[i]),
                connectgaps=True,
            ))

            # endpoints
            fig.add_trace(go.Scatter(
                x=[x_data[i][0], x_data[i][-1]],
                y=[y_data[i][0], y_data[i][-1]],
                mode='markers',
                marker=dict(color=colors[i], size=mode_size[i])
            ))


        ############## Data plotting phase completed

        ############ phase 5:- FIgure layout phase

        fig.update_layout(
            xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82, 82, 82)',
                ),
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
            ),
            autosize=False,
            margin=dict(
                autoexpand=False,
                l=100,
                r=20,
                t=110,
            ),
            showlegend=False,
            plot_bgcolor='white'
        )

########### layout phase completed

### Phase 6:- Anotation phase
        annotations = []

        # Adding labels
        for y_trace, label, color in zip(y_data, labels, colors):
            # labeling the left_side of the plot
            annotations.append(dict(xref='paper', x=0.1, y=y_trace[0],
                                          xanchor='right', yanchor='middle',
                                          text=label + ' {}%'.format(y_trace[0]),
                                          font=dict(family='Arial',
                                                    size=16),
                                          showarrow=False))
            # labeling the right_side of the plot
            annotations.append(dict(xref='paper', x=0.9, y=y_trace[11],
                                          xanchor='left', yanchor='middle',
                                          text='{}%'.format(y_trace[11]),
                                          font=dict(family='Arial',
                                                    size=16),
                                          showarrow=False))
        # Title
        annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                      xanchor='left', yanchor='bottom',
                                      text='Readings of the Smart Fasal',
                                      font=dict(family='Arial',
                                                size=30,
                                                color='rgb(37,37,37)'),
                                      showarrow=False))
        # Source
        annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,
                                      xanchor='center', yanchor='top',
                                      text='Waspmote device',
                                      font=dict(family='Arial',
                                                size=12,
                                                color='rgb(150,150,150)'),
                                      showarrow=False))

        fig.update_layout(annotations=annotations)

        fig.show()
### Anmotation phase completed. 
##########################################################################################################








#

############### to access the data-files
    def ftp_login(request):
        url='ftp.smartfasal.in'
        username = 'testuser@smartfasal.in'
        pwd = 'fasal@thapar'

        import os
        from smartfasal_project.settings import BASE_DIR
        STATIC_ROOT = os.path.join(BASE_DIR, 'static')
        path = STATIC_ROOT
        os.chdir(path)
        #file_path = os.path.join(BASE_DIR, path)
        import ftplib
        ftp = ftplib.FTP(url, username, pwd)
        files = ftp.dir()
        ftp.cwd("/")
        filename = 'S_AgriB.csv'
        my_file = open(filename, 'wb') # Open a local file to store the downloaded file
        ftp.retrbinary('RETR ' + filename, my_file.write) # Enter
        ftp.quit() # Terminate the FTP connection
        my_file.close() # Close the local file you had opened for do
        return HttpResponse("Data fetched\n plz check static folder")




    def visualising(request):
        return render(request,'visualising.html')
