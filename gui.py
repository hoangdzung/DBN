from tkinter import *
import tkinter.filedialog
import ttk
import numpy as np
import pandas as pd
import sys
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
from openpyxl.drawing.text import TextField

def positive_normal_random_gen(mu = 15,sigma=30, size=1000):
    count = 0
    ran_list = []
    while (count < size):
        a = np.random.normal(mu, sigma)
        if (a >= 0):
            ran_list.append(int(a))
            count = count + 1
            if (count >= size):
                break
    # count = np.zeros(300)
    # for a in ran_list:
    #     count[a] = count[a]+1
    # plt.figure(1)
    # plt.plot(count)
    return np.array(ran_list)

class Application(Frame):
    def choosefile(self):
        self.file_path = tkinter.filedialog.askopenfilename()
        self.datafile_label["text"] = self.file_path

    def process(self):
        # print self.dpq_box.get()
        # print self.c_box.get()
        # Check input
        if not hasattr(self, 'file_path'):
            print ("chua chon file")
            exit()
        pr = {}
        # 'very low', 'low', 'medium', 'high', 'very high'
        # print type(self.dpq_box.get())
        if self.dpq_box.get() == 'unknown':
            if 'DPQ' in pr.keys():
                del pr['DPQ']
        elif self.dpq_box.get() == 'very low':
            pr['DPQ'] = 0
        elif self.dpq_box.get() == 'low':
            pr['DPQ'] = 1
        elif self.dpq_box.get() == 'medium':
            pr['DPQ'] = 2
        elif self.dpq_box.get() == 'high':
            pr['DPQ'] = 3
        elif self.dpq_box.get() == 'very high':
            pr['DPQ'] = 4
        else:
            pass

        if self.c_box.get() == 'unknown':
            if 'C' in pr.keys():
                del pr['C']
        elif self.c_box.get() == 'very low':
            pr['C'] = 0
        elif self.c_box.get() == 'low':
            pr['C'] = 1
        elif self.c_box.get() == 'medium':
            pr['C'] = 2
        elif self.c_box.get() == 'high':
            pr['C'] = 3
        elif self.c_box.get() == 'very high':
            pr['C'] = 4
        else:
            pass

        if self.tq_box.get() == 'unknown':
            if 'TQ' in pr.keys():
                del pr['TQ']
        elif self.tq_box.get() == 'very low':
            pr['TQ'] = 0
        elif self.tq_box.get() == 'low':
            pr['TQ'] = 1
        elif self.tq_box.get() == 'medium':
            pr['TQ'] = 2
        elif self.tq_box.get() == 'high':
            pr['TQ'] = 3
        elif self.tq_box.get() == 'very high':
            pr['TQ'] = 4
        else:
            pass

        if self.ou_box.get() == 'unknown':
            if 'OU' in pr.keys():
                del pr['OU']
        elif self.ou_box.get() == 'very low':
            pr['OU'] = 0
        elif self.ou_box.get() == 'low':
            pr['OU'] = 1
        elif self.ou_box.get() == 'medium':
            pr['OU'] = 2
        elif self.ou_box.get() == 'high':
            pr['OU'] = 3
        elif self.ou_box.get() == 'very high':
            pr['OU'] = 4
        else:
            pass

        print (pr)

        # Get environment variables

        # priority
        # pr={
        #     'DPQ':4
        # }

        data = pd.read_csv(self.file_path)  # "fisrm.csv"
        data_size = len(data)

        # xac dinh phan phoi cho cac node o input layer
        # cpd_tq = TabularCPD(variable='TQ', variable_card=5, values=[[0.2, 0.2,0.2,0.2,0.2]])
        # cpd_c = TabularCPD(variable='C', variable_card=5, values=[[0.2, 0.2,0.2,0.2,0.2]])
        # cpd_dpq = TabularCPD(variable='DPQ', variable_card=5, values=[[0.2, 0.2,0.2,0.2,0.2]])
        # cpd_ou = TabularCPD(variable='OU', variable_card=5, values=[[0.2, 0.2,0.2,0.2,0.2]])

        # dinh nghia cau truc mang bayes

        # TQ: Test quality, DPQ: design process quality, C: complexity, DI: defects inserted, DFT: defects found intesting
        # RD: desidual defects, OU: operational usage, DFO: defects found in operation
        model = BayesianModel(
            [('TQ', 'DFT'), ('DPQ', 'DI'), ('C', 'DI'), ('DI', 'DFT'), ('DI', 'RD'), ('DFT', 'RD'), ('RD', 'DFO'),
             ('OU', 'DFO')])

        model.fit(data, estimator_type=BayesianEstimator, prior_type="BDeu",
                  equivalent_sample_size=10)  # default equivalent_sample_size=5
        # for cpd in model.get_cpds():
        #     print(cpd)
        # print model.get_cpds()[2]


        infer = VariableElimination(model)
        DI_distribution = infer.query(['DI'])['DI'].values
        # DI_distribution = infer.query(['DI'], evidence={'DPQ': 2, 'C': 3, 'TQ': 4,'OU':1})['DI'].values
        max_DI = np.argmax(DI_distribution)
        print (max_DI)
        print (infer.query(['DPQ'])['DPQ'])
        print (model.get_cpds()[1])

        plt.figure(1)
        # plt.subplot(4, 2, 2)
        # plt.plot(DI_distribution)
        # plt.title("defects inserted")
        # plt.xlabel('number')
        # plt.ylabel('probability')


        nodes = ['DPQ', 'C', 'TQ', 'DI', 'DFT', 'RD', 'OU', 'DFO']
        Distribution = {}
        # print np.sign(0)
        for key in pr.keys():
            Distribution[key] = [1 - abs(np.sign(pr[key] - i)) for i in range(5)]
            nodes.remove(key)

        for key in nodes:
            Distribution[key] = infer.query([key], evidence=pr)[key].values
        # Distribution['C']=infer.query(['C'], evidence=pr)['C'].values
        # Distribution['DI']=infer.query(['DI'], evidence=pr)['DI'].values
        # Distribution['DFT']=infer.query(['DFT'], evidence=pr)['DFT'].values
        # Distribution['RD']=infer.query(['RD'], evidence=pr)['RD'].values
        # Distribution['DFO']=infer.query(['DFO'], evidence=pr)['DFO'].values
        # Distribution['DPQ']=[0,0,0,0,1]
        # Distribution['TQ']=[0,0,0,0,1]
        # Distribution['OU']=[0,0,0,0,1]
        # Distribution = infer.query(['DPQ','C','TQ','DI','DFT','RD','OU','DFO'], evidence={'DPQ': 2, 'TQ': 4,'OU':1})

        plt.subplot(4, 2, 1)
        plt.bar([1, 2, 3, 4, 5], Distribution['DPQ'])
        plt.xticks([1.5, 2.5, 3.5, 4.5, 5.5], ['very low', 'low', 'medium', 'high', 'very high'])
        plt.title("design process quality")
        # plt.xlabel('number')
        plt.ylabel('probability')

        plt.subplot(4, 2, 2)
        plt.bar([1, 2, 3, 4, 5], Distribution['C'])
        plt.xticks([1.5, 2.5, 3.5, 4.5, 5.5], ['very low', 'low', 'medium', 'high', 'very high'])
        plt.title("complexity")
        # plt.xlabel('number')
        plt.ylabel('probability')

        plt.subplot(4, 2, 3)
        plt.bar([1, 2, 3, 4, 5], Distribution['TQ'])
        plt.xticks([1.5, 2.5, 3.5, 4.5, 5.5], ['very low', 'low', 'medium', 'high', 'very high'])
        plt.title("Test quality")
        # plt.xlabel('number')
        plt.ylabel('probability')

        plt.subplot(4, 2, 4)
        plt.plot(Distribution['DI'])
        plt.title("defects inserted")
        # plt.xlabel('number')
        plt.ylabel('probability')

        plt.subplot(4, 2, 5)
        plt.plot(Distribution['DFT'])
        plt.title("defects found intesting")
        # plt.xlabel('number')
        plt.ylabel('probability')

        plt.subplot(4, 2, 6)
        plt.plot(Distribution['RD'])
        plt.title("desidual defects")
        # plt.xlabel('number')
        plt.ylabel('probability')

        plt.subplot(4, 2, 7)
        plt.bar([1, 2, 3, 4, 5], Distribution['OU'])
        plt.xticks([1.5, 2.5, 3.5, 4.5, 5.5], ['very low', 'low', 'medium', 'high', 'very high'])
        plt.title("operational usage")
        # plt.xlabel('number')
        plt.ylabel('probability')

        plt.subplot(4, 2, 8)
        plt.plot(Distribution['DFO'])
        plt.title("defects found in operation")
        # plt.xlabel('number')
        plt.ylabel('probability')

        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def createWidgets(self):
        pad_x = 5
        pad_y = 5
        self.firstlabel = Label(self)
        self.firstlabel["text"] = "Choose_file:__",
        # self.text_datafile["command"] = self.say_hi

        # self.firstlabel.grid(row=0, column=1, padx=pad_x, pady=pad_y, sticky=W)

        self.datafile_label = Label(self)
        self.datafile_label["text"] = "no_file",
        self.datafile_label.grid(row=0, column=1, padx=pad_x, pady=pad_y,columnspan=3, sticky=W)

        self.choosefilebutton = Button(self)
        self.choosefilebutton["text"] = "Choose_data_file",
        self.choosefilebutton["command"] = self.choosefile
        self.choosefilebutton.grid(row=1, column=1, padx=pad_x, pady=pad_y, sticky=W)

        self.dpq_label = Label(self)
        self.dpq_label["text"] = "design_process_quality",
        self.dpq_label.grid(row=2, column=1, padx=pad_x, pady=pad_y, sticky=W)

        # self.dpq_entry = Entry(self, width=10)
        # self.dpq_entry.grid(row=1, column=3, padx=pad_x, pady=pad_y, sticky=W)
        self.dpq_box_value = StringVar()
        self.dpq_box = ttk.Combobox(self, textvariable=self.dpq_box_value)
        self.dpq_box['values'] = ('unknown','very low', 'low', 'medium', 'high', 'very high')
        self.dpq_box.current(0)
        self.dpq_box.grid(row=2, column=2, padx=pad_x, pady=pad_y, sticky=W)

        self.c_label = Label(self)
        self.c_label["text"] = "Complexity",
        self.c_label.grid(row=2, column=3, padx=pad_x, pady=pad_y, sticky=W)

        # self.c_entry = Entry(self, width=10)
        # self.c_entry.grid(row=1, column=5, padx=pad_x, pady=pad_y, sticky=W)
        self.c_box_value = StringVar()
        self.c_box = ttk.Combobox(self, textvariable=self.c_box_value)
        self.c_box['values'] = ('unknown','very low', 'low', 'medium', 'high', 'very high')
        self.c_box.current(0)
        self.c_box.grid(row=2, column=4, padx=pad_x, pady=pad_y, sticky=W)

        self.tq_label = Label(self)
        self.tq_label["text"] = "Test_quality",
        self.tq_label.grid(row=2, column=5, padx=pad_x, pady=pad_y, sticky=W)

        # self.tq_entry = Entry(self, width=10)
        # self.tq_entry.grid(row=1, column=7, padx=pad_x, pady=pad_y, sticky=W)

        self.tq_box_value = StringVar()
        self.tq_box = ttk.Combobox(self, textvariable=self.tq_box_value)
        self.tq_box['values'] = ('unknown','very low','low','medium', 'high','very high')
        self.tq_box.current(0)
        self.tq_box.grid(row=2, column=6, padx=pad_x, pady=pad_y, sticky=W)

        self.ou_label = Label(self)
        self.ou_label["text"] = "operational_usage",
        self.ou_label.grid(row=2, column=7, padx=pad_x, pady=pad_y, sticky=W)

        # self.tq_entry = Entry(self, width=10)
        # self.tq_entry.grid(row=1, column=7, padx=pad_x, pady=pad_y, sticky=W)

        self.ou_box_value = StringVar()
        self.ou_box = ttk.Combobox(self, textvariable=self.ou_box_value)
        self.ou_box['values'] = ('unknown', 'very low', 'low', 'medium', 'high', 'very high')
        self.ou_box.current(0)
        self.ou_box.grid(row=2, column=8, padx=pad_x, pady=pad_y, sticky=W)

        self.processbotton = Button(self)
        self.processbotton["text"] = "Process",
        self.processbotton["command"] = self.process
        self.processbotton.grid(row=3, column=1, padx=pad_x, pady=pad_y, sticky=W)

        self.QUIT = Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"] = "red"
        self.QUIT["command"] = self.quit
        self.QUIT.grid(row=3, column=2, padx=pad_x, pady=pad_y, sticky=W)

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

root = Tk()
app = Application(master=root)
app.mainloop()
root.destroy()
