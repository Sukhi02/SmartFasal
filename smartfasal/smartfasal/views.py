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


    def Real_time_plot(request):
        ######### Phase 1:- Data aloocating and defining
        import plotly.graph_objects as go
        import plotly
        import numpy as np
        import plotly.express as px

        ########### phase :: Accessing the downloaded file (Data1.csv)
        import os
        from smartfasal_project.settings import BASE_DIR
        STATIC_ROOT = os.path.join(BASE_DIR, 'static')
        path = STATIC_ROOT
        os.chdir(path)
        import pandas as pd
        filename = "data1.csv"
        #Local_data_smartfasal = pd.read_csv(filename, names=['Timestamp', 'S_M_10cm','S_M_45cm','S_M_80cm', 'Temperature', 'Humidity', 'Pressure', 'Luxes'])
        Local_data_smartfasal = pd.read_csv(filename)
        import plotly.express as px

        df = pd.read_csv('datetime.csv')


        x_Time = df['Time']
        y0 = df['SM10']
        y1 = df['SM45']
        y2 = df['SM80']
        y3 = df['Temp']
        y4 = df['Humd']
        y5 = df['LMNS']
        y6 = df['PRSR']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_Time, y=y0, mode='lines', name='SM @ 10 cm',
                                marker = dict(color = 'rgba(255,0,0,0.8)')))
        fig.add_trace(go.Scatter(x=x_Time, y=y1, mode='lines', name='SM @ 45 cm',
                                marker = dict(color = 'rgba(0,255,0,0.8)')))
        fig.add_trace(go.Scatter(x=x_Time, y=y2, mode='lines', name='SM @ 80 cm',
                                marker = dict(color = 'rgba((26, 102, 255,0.8)')))
        fig.add_trace(go.Scatter(x=x_Time, y=y3, mode='lines', name='Temperature',
                                marker = dict(color = 'rgba(204, 0, 204, 0.8)')))
        fig.add_trace(go.Scatter(x=x_Time, y=y4, mode='lines', name='Humidity',
                                marker = dict(color = 'rgba(0, 153, 51, 0.8)')))
        fig.add_trace(go.Scatter(x=x_Time, y=y5, mode='lines', name='Luminisity',
                                marker = dict(color = 'rgba(0, 0, 204, 0.8)')))
        fig.add_trace(go.Scatter(x=x_Time, y=y6, mode='lines', name='Pressure',
                                marker = dict(color = 'rgba(80, 26, 80, 0.8)')))
        fig.show()

        import chart_studio
        username = 'sukhi02' # your usernam
        api_key = 'VQ5pvk3TMJi50tDGdWne' # your api key - go to profile > settings > regenerate keychart_studio.tools.set_credentials_file(username=username, api_key=api_key)
        chart_studio.tools.set_credentials_file(username=username, api_key=api_key)


        import chart_studio.plotly as csp
        csp.plot(fig,   showLink= 'false');

        #Plotly.plot(divid, data, layout, {showLink: false})
        #plt_div = plot(fig, output_type='div', include_plotlyjs=False)
        #return render(request,'www.http://smartfasal.in/wp/?page_id=277')
        return HttpResponse("Prcoessed completed")





    ######################################################################################################
        """def jslearning(request):

            ######### Phase 1:- Data aloocating and defining
            import plotly.graph_objects as go
            import plotly
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
                                              text=label + ' {}'.format(y_trace[0]),
                                              font=dict(family='Arial',
                                                        size=16),
                                              showarrow=False))
                # labeling the right_side of the plot
                annotations.append(dict(xref='paper', x=0.9, y=y_trace[11],
                                              xanchor='left', yanchor='middle',
                                              text='{}'.format(y_trace[11]),
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




            import chart_studio
            username = 'sukhi02' # your usernam
            api_key = 'VQ5pvk3TMJi50tDGdWne' # your api key - go to profile > settings > regenerate keychart_studio.tools.set_credentials_file(username=username, api_key=api_key)
            chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

            import chart_studio.plotly as py
            #py.plot(fig, filename = 'SmartFasal_Readings')

            #return render(request,'www.http://smartfasal.in/wp/?page_id=277')
            return HttpResponse("Prcoessed completed")"""





    ### Anmotation phase completed.
    ##########################################################################################################


class Plots:
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
        return Plots.real_plot(request)

    def real_plot(request):
        print("Accesing the FTP")
        import pandas as pd
        import numpy as np
        import os
        from smartfasal_project.settings import BASE_DIR
        STATIC_ROOT = os.path.join(BASE_DIR, 'static')
        path = STATIC_ROOT
        os.chdir(path)

        print(" Step 6: Library imported")

        #Load the CSV file
        print(" Step 7: Load the CSV file")

        data_file = pd.read_csv("S_AgriB.csv")

        print(" >>>     File loaded succesfiully")
        print(data_file.head())

        ############################################################

        print("##################################################")
        print(" Part 3: Modifying the downloaded file")
        print("##################################################")

        print(" Step 8: Load the dataset")

        cols_names=['Timestamp', 'S_M_10cm','S_M_45cm','S_M_80cm', 'Temperature', 'Humidity', 'Pressure', 'Luxes', 'Battery','Readings','a', 'b', 'c']

        data_file.columns = cols_names

        data_file.to_csv("s.csv")
        print("         Column name upgraded")

        print("         Convert epochs to date-time")

        date_file = pd.read_csv("s.csv")
        date_file = data_file.drop(['S_M_10cm','S_M_45cm','S_M_80cm', 'Temperature', 'Humidity', 'Pressure', 'Luxes', 'Battery','Readings','a', 'b', 'c'], axis = 1)
        cols_names=['Time']
        date_file.columns = cols_names
        date_file.to_csv("date_file.csv")

    ###########################################################
    ##########################################################
        def dateparse(time_in_secs):
           import datetime
           time_in_secs = time_in_secs
           return datetime.datetime.fromtimestamp(float(time_in_secs))

        dtype= {"Time": float, "Value":float}
        date_time = pd.read_csv("date_file.csv", dtype=dtype, parse_dates=["Time"], date_parser=dateparse)
        print(date_time)
        print("  >> Successfully converted epochs to the date_time_format")
        #######################
        #######################
        #######################
        date_time = date_time['Time']

        data_file.index = date_time
        data_file = data_file.drop(['Timestamp','Battery', 'Readings', 'a', 'b', 'c'], axis =1)
        print(" >>   Data cleaned successfully")

        ######################
        print("##################################################")
        print("    Part 4: Import 400 observations")
        print("##################################################")

        last_rows = data_file.iloc[-400:, 0:7]
        date_time = last_rows.index
        del date_file
        print("last 400 observations are succssfully imported")

        print("step 9: plot the dataset")
        import matplotlib.pyplot as plt

        #plt.plot(last_rows, label = 'Predicted')
        #plt.xlabel('Time')
        #plt.ylabel('Lumisity')
        #plt.title('Forecasted Lum')
        #plt.legend()
        #plt.savefig('Lum ARIMA FULL')
        #plt.show()
        print("Succssfully plotted")

        del dtype

        ######################
        print("##################################################")
        print("    Part 5:  Scalling the dataset")
        print("##################################################")

        sm1 = last_rows.iloc[:, 0:1].values
        sm2 = last_rows.iloc[:, 1:2].values
        sm3 = last_rows.iloc[:, 2:3].values
        Temp= last_rows.iloc[:, 3:4].values
        Humd= last_rows.iloc[:, 4:5].values
        Prsr= last_rows.iloc[:, 5:6].values
        Lmns= last_rows.iloc[:, 6:7].values

        sm1 = sm1/100
        sm2 = sm2/100
        sm3 = sm3/100
        Prsr= Prsr/1000
        Lmns= Lmns/100

        sm1 = pd.DataFrame(sm1, index = date_time, columns=['SM10'])
        sm2 = pd.DataFrame(sm2, index = date_time, columns=['SM45'])
        sm3 = pd.DataFrame(sm3, index = date_time, columns=['SM80'])
        Temp= pd.DataFrame(Temp, index = date_time, columns=['Temp'])
        Humd= pd.DataFrame(Humd, index = date_time, columns=['Humd'])
        Prsr= pd.DataFrame(Prsr, index = date_time, columns=['Prsr'])
        Lmns= pd.DataFrame(Lmns, index = date_time, columns=['Lmns'])

        last_rows = pd.concat([sm1, sm2, sm3, Temp, Humd, Prsr, Lmns], axis =1)
        last_rows.to_csv("Last_rows.csv")


        del sm1, sm2, sm3, Humd, Lmns, Temp, Prsr,  cols_names, data_file, date_time
        print (" Cache Cleared")
        #plt.plot(last_rows)
        ##plt.xlabel('Time')
        #plt.ylabel('Lumisity')
        #plt.title('Forecasted Lum')
        #plt.legend()
        #plt.savefig(' new Plot')
        #plt.show()
        print("Succssfully plotted")
        print(" Memory Empty")
        del last_rows
        return HttpResponse("D O N E ")

########        ############### to access the data-files
    #############################################################################



#############################################################################



def visualising(request):
    return render(request,'visualising.html')
