from django.db import models
import ftplib
import matplotlib.pyplot as plt
import pandas as pd

class FTP_session_model(models.Model):
    def FTP_On(self):
        FTP_URL = "ftp://ftp.smartfasal.in/"
        login = "testuser@smartfasal.in"
        pswd= "fasal@thapar"
        return(FTP_URL)





    def FTP_On_test(self):

        url='ftp.smartfasal.in'
        username = 'testuser@smartfasal.in'
        pwd = 'fasal@thapar'

        import matplotlib.pyplot as plt
        import pandas as pd

        filename = 'S_AgriB.csv'
        print('file name is ' + filename)
        import os
        #os.chdir("C:/Users/Raju/Downloads")

        data = pd.read_csv(filename, names=['Timestamp', 'S_M_10cm','S_M_45cm','S_M_80cm', 'Temperature', 'Humidity', 'Pressure', 'Luxes', 'Battery','Readings','Day', 'Date', 'Time'])
        # line 1 points

        x1 = data["Timestamp"]


        y4 =  data["Temperature"]
        plt.plot(x1, y4, label = "Temperature")
        plt.ylabel('Atmospheric Temperature')
        plt.xlabel('Time')
        plt.title('Temp of the day')
        plt.legend()
        #plt.savefig("Temperature")
        plot= plt.show()

        return plot
