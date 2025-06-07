import tkinter as tk
from PIL import Image, ImageTk
import pickle
import numpy as np

print("test")

# imports
with open('.\\saved_models\\testset.pkl', 'rb') as file:
    testset = pickle.load(file)
print('testset loaded')

with open('.\\saved_models\\AIRS_A.pkl', 'rb') as file:
    AIRS_A_model = pickle.load(file)
with open('.\\saved_models\\AIRS_B.pkl', 'rb') as file:
    AIRS_B_model = pickle.load(file)
with open('.\\saved_models\\AIRS_C.pkl', 'rb') as file:
    AIRS_C_model = pickle.load(file)

print('AIRS models loaded')

with open('.\\saved_models\\RF_A.pkl', 'rb') as file:
    RF_A_model = pickle.load(file)
with open('.\\saved_models\\RF_B.pkl', 'rb') as file:
    RF_B_model = pickle.load(file)
with open('.\\saved_models\\RF_C.pkl', 'rb') as file:
    RF_C_model = pickle.load(file)

print('RF models loaded')

with open('.\\saved_models\\Hybrid_RF_A.pkl', 'rb') as file:
    Hybrid_A_model = pickle.load(file)
with open('.\\saved_models\\Hybrid_RF_B.pkl', 'rb') as file:
    Hybrid_B_model = pickle.load(file)
with open('.\\saved_models\\Hybrid_RF_C.pkl', 'rb') as file:
    Hybrid_C_model = pickle.load(file)

print('Hybrid models loaded')

selectedSample = None
selectedClass = None

map_C = {'Portmap' : 0,
'NetBIOS' : 1,
'DrDoS_NetBIOS' : 2,
'DrDoS_DNS' : 3,
'DrDoS_LDAP' : 4,
'LDAP' : 5,
'DrDoS_SNMP' : 6,
'UDP' : 7,
'DrDoS_UDP' : 8,
'UDP-lag' : 9,
'UDPLag' : 9,
'Benign' : 10,
'DrDoS_MSSQL' : 11,
'MSSQL' : 12,
'WebDDoS' : 13,
'Syn' : 14,
'TFTP' : 15,
'DrDoS_NTP' : 16,
}
map_B = {'Portmap' : 0,
'NetBIOS' : 1,
'DrDoS_NetBIOS' : 1,
'DrDoS_NTP' : 11,
'DrDoS_LDAP' : 3,
'LDAP' : 3,
'DrDoS_SNMP' : 4,
'UDP' : 5,
'UDP-lag' : 5,
'UDPLag' : 5,
'DrDoS_UDP' : 5,
'Benign' : 6,
'DrDoS_MSSQL' : 7,
'MSSQL' : 7,
'WebDDoS' : 8,
'Syn' : 9,
'TFTP' : 10,
'DrDoS_DNS' : 2}
map_A = {'Portmap' : 0,
'DrDoS_NTP' : 0,
'NetBIOS' : 0,
'DrDoS_LDAP' : 0,
'UDP' : 0,
'DrDoS_SNMP' : 0,
'Benign' : 1,
'DrDoS_NetBIOS' : 0,
'DrDoS_MSSQL' : 0,
'MSSQL' : 0,
'WebDDoS' : 0,
'Syn' : 0,
'LDAP' : 0,
'TFTP' : 0,
'UDP-lag' : 0,
'UDPLag' : 0,
'DrDoS_UDP' : 0,
'DrDoS_DNS' : 0,
'Attack' : 0}

reverse_map_C = {
    0 : 'Portmap',
    1 : 'NetBIOS',
    2 : 'DrDoS_NetBIOS',
    3 : 'DrDoS_DNS',
    4 : 'DrDoS_LDAP',
    5 : 'LDAP',
    6 : 'DrDoS_SNMP',
    7 : 'UDP',
    8 : 'DrDoS_UDP',
    9 : 'UDP-lag',
    10: 'Benign',
    11: 'DrDoS_MSSQL',
    12: 'MSSQL',
    13: 'WebDDoS',
    14: 'Syn',
    15: 'TFTP',
    16: 'DrDoS_NTP'
}
reverse_map_B = {
    0 : 'Portmap',
    1 : 'NetBIOS',
    2 : 'DrDoS_DNS',
    3 : 'LDAP',
    4 : 'DrDoS_SNMP',
    5 : 'UDP',
    6 : 'Benign',
    7 : 'MSSQL',
    8 : 'WebDDoS',
    9 : 'Syn',
    10 : 'TFTP',
    11 : 'DrDoS_NTP',
}
reverse_map_A = {0: 'Attack', 1: 'Benign'}


# UI Functions
def SectionChange(newSection):
    train_section_btn.configure(bg=bg_color_400, relief='flat', fg=text_color_500)
    test_section_btn.configure(bg=bg_color_400, relief='flat', fg=text_color_500)
    stat_section_btn.configure(bg=bg_color_400, relief='flat', fg=text_color_500)
    match newSection:
        case 'train':
            train_section_btn.configure(bg=bg_color_500, relief='groove', fg=text_color_100)
            mainContainer_train.tkraise()
        case 'test':
            test_section_btn.configure(bg=bg_color_500, relief='groove', fg=text_color_100)
            mainContainer_test.tkraise()
        case 'stat':
            stat_section_btn.configure(bg=bg_color_500, relief='groove', fg=text_color_100)
            mainContainer_stat.tkraise()


def DisplayFigure(model, algorithm, category):
    # must be written like this
    # category_algorithm_Model_model.png

    if model == 'all':
        # metrics part
        image_path = f'.\\figures\\comparaison_{algorithm}.png'
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            return
        image = image.resize((700, 450))
        image = ImageTk.PhotoImage(image)
    else:
        # confusion matrices part
        image_path = f'.\\figures\\{category}_{algorithm}_Model_{model}.png'
        # image_path = f'.\\figures\\confusion_matrix_{algorithm}_Model_{model}.png'
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            return
        image = image.resize((600, 600))
        image = ImageTk.PhotoImage(image)

    # update the image in the label
    imageFrame.config(image=image)
    imageFrame.image = image



def selectRandomSample():
    random_index, sample, sample_class = getRandomSample()
    # update ui
    RandomSampleIndexLabel.config(text=f"Échantillon {random_index}")
    RandomSampleClassLabel.config(text=sample_class)

    # set global variables
    global selectedSample, selectedClass
    selectedSample = sample
    selectedClass = sample_class

    predictResults()

def predictResults():

    pred_A_AIRS = reverse_map_A[AIRS_A_model.classify([selectedSample])]
    pred_B_AIRS = reverse_map_B[AIRS_B_model.classify([selectedSample])]
    pred_C_AIRS = reverse_map_C[AIRS_C_model.classify([selectedSample])]

    pred_A_RF = reverse_map_A[RF_A_model.predict([selectedSample])[0]]
    pred_B_RF = reverse_map_B[RF_B_model.predict([selectedSample])[0]]
    pred_C_RF = reverse_map_C[RF_C_model.predict([selectedSample])[0]]

    pred_A_HYB = reverse_map_A[Hybrid_A_model.predict(toMCPool([selectedSample], [map_A[selectedClass]], AIRS_A_model))[0]]
    pred_B_HYB = reverse_map_B[Hybrid_B_model.predict(toMCPool([selectedSample], [map_B[selectedClass]], AIRS_B_model))[0]]
    pred_C_HYB = reverse_map_C[Hybrid_C_model.predict(toMCPool([selectedSample], [map_C[selectedClass]], AIRS_C_model))[0]]


    global A_AIRS, B_AIRS, C_AIRS, A_RF, B_RF, C_RF, A_HYB, B_HYB, C_HYB
    A_AIRS.config(text=pred_A_AIRS, bg=good_color if reverse_map_A[map_A[pred_A_AIRS]] == reverse_map_A[map_A[selectedClass]] else bad_color)
    B_AIRS.config(text=pred_B_AIRS, bg=good_color if reverse_map_B[map_B[pred_B_AIRS]] == reverse_map_B[map_B[selectedClass]] else bad_color)
    C_AIRS.config(text=pred_C_AIRS, bg=good_color if reverse_map_C[map_C[pred_C_AIRS]] == reverse_map_C[map_C[selectedClass]] else bad_color)
    A_RF.config(text=pred_A_RF, bg=good_color if reverse_map_A[map_A[pred_A_RF]] == reverse_map_A[map_A[selectedClass]] else bad_color)
    B_RF.config(text=pred_B_RF, bg=good_color if reverse_map_B[map_B[pred_B_RF]] == reverse_map_B[map_B[selectedClass]] else bad_color)
    C_RF.config(text=pred_C_RF, bg=good_color if reverse_map_C[map_C[pred_C_RF]] == reverse_map_C[map_C[selectedClass]] else bad_color)
    A_HYB.config(text=pred_A_HYB, bg=good_color if reverse_map_A[map_A[pred_A_HYB]] == reverse_map_A[map_A[selectedClass]] else bad_color)
    B_HYB.config(text=pred_B_HYB, bg=good_color if reverse_map_B[map_B[pred_B_HYB]] == reverse_map_B[map_B[selectedClass]] else bad_color)
    C_HYB.config(text=pred_C_HYB, bg=good_color if reverse_map_C[map_C[pred_C_HYB]] == reverse_map_C[map_C[selectedClass]] else bad_color)



# Useful Functions
def getRandomSample():
    random_index = np.random.randint(0, len(testset))
    sample = testset[random_index, :-1]
    sample_class = reverse_map_C[testset[random_index, -1]]
    return random_index, sample, sample_class

def toMCPool(x_test, y_test, model):
    new_x_test = []
    for antigene, _class in zip(x_test, y_test):
        # finding the mc cell with highest affinity with antigene
        best_aff = 0.0
        for cell in model.MC_POOL.get(_class):
            aff = model.affinity(antigene, cell.vector)
            if aff > best_aff:
                best_aff = aff
                best_aff_vector = cell.vector
        
        # adding the best aff vector to the new xtest
        new_x_test.append(best_aff_vector)
    return np.array(new_x_test)

# def affinity(antigene, cell_vector):
#     return 1 / (1 + np.linalg.norm(antigene - cell_vector))


# constants
bg_color_500 = '#2c2c2c'
bg_color_400 = "#333333"
bg_color_200 = "#444444"

good_color = "#207220"
bad_color = "#722020"

text_color_1000 = 'black'
text_color_500 = "#777777"
text_color_100 = 'white'

borderThinkness = 3

fontFamily = 'Helvetica'
bigText = (fontFamily, 20, 'bold')
normalText = (fontFamily, 14)
normalText_bold = (fontFamily, 14, 'bold')
smallText = (fontFamily, 12)
verysmallText = (fontFamily, 10)

sideBarWidth = 200


# root Window
root = tk.Tk()
root.title('DDoS Classification')
root.geometry('1220x720')
root.resizable(False, False)
root.configure(background=bg_color_500)


# Side Bar
sideBar = tk.Frame(root, width=sideBarWidth, bg=bg_color_400)
sideBar.pack_propagate(False)
sideBar.pack(side='left', fill='y')

# Sidebar Layout
tk.Frame(sideBar, bg=bg_color_400).pack(expand=True)   # spacer
sideBarFrame = tk.Frame(sideBar, bg=bg_color_400)
sideBarFrame.pack()
tk.Frame(sideBar, bg=bg_color_400).pack(expand=True)   # spacer

# Sidebar Buttons
train_section_btn = tk.Button(sideBarFrame, text='Entrainement', font=('Helvetica', 18), relief='flat', justify='left', fg=text_color_100, bg=bg_color_400, command=lambda : SectionChange('train'))
test_section_btn = tk.Button(sideBarFrame, text='Evaluation', font=('Helvetica', 18), relief='flat', justify='left', fg=text_color_100, bg=bg_color_400, command=lambda : SectionChange('test'))
stat_section_btn = tk.Button(sideBarFrame, text='Comparaison', font=('Helvetica', 18), relief='flat', justify='left', fg=text_color_100, bg=bg_color_400, command=lambda : SectionChange('stat'))
train_section_btn.pack(padx=5, pady=10, fill='x')
test_section_btn.pack(padx=5, pady=10, fill='x')
stat_section_btn.pack(padx=5, pady=10, fill='x')







# Main Container
mainContainer = tk.Frame(root, bg=bg_color_500)
mainContainer.pack(fill='both', expand=True)

# train section
mainContainer_train = tk.Frame(mainContainer, bg=bg_color_500)
mainContainer_train.place(relx=0, rely=0, relheight=1, relwidth=1)
tk.Label(mainContainer_train, text='Entrainement des models', fg=text_color_100, bg=bg_color_500, font=bigText).pack(padx=10, pady=40, fill='x')

# models container layout
modelsCont = tk.Frame(mainContainer_train, bg=bg_color_500)
modelsCont.pack(padx=10, pady=70)
modelA_train = tk.Frame(modelsCont, bg=bg_color_500, relief='groove', highlightbackground=bg_color_200, highlightthickness=borderThinkness)
modelB_train = tk.Frame(modelsCont, bg=bg_color_500, relief='groove', highlightbackground=bg_color_200, highlightthickness=borderThinkness)
modelC_train = tk.Frame(modelsCont, bg=bg_color_500, relief='groove', highlightbackground=bg_color_200, highlightthickness=borderThinkness)
modelsCont.grid_rowconfigure(0, weight=1)
for col in range(3): modelsCont.grid_columnconfigure(col, weight=1)
modelA_train.grid(row=0, column=0, sticky='nsew', padx=(5,10))
modelB_train.grid(row=0, column=1, sticky='nsew', padx=(10,10))
modelC_train.grid(row=0, column=2, sticky='nsew', padx=(10,5))

# models information
modelsInfo = {
    'A': {
        'frame': modelA_train,
        'name': 'Model A',
        'description': "classifie entre les flux DDoS et les flux normaux",
        'classNumber': 2,
        'classes' : ['Begnin', 'Attack'],
    },
    'B': {
        'frame': modelB_train,
        'name': 'Model B',
        'description': "classifie les flux suivant les patternes d'attaques généraux",
        'classNumber': 12,
        'classes' : ['Portmap', 'NetBIOS', 'DrDoS_DNS', 'LDAP', 'SNMP', 'UDP', 'Benign', 'MSSQL', 'WebDDoS', 'Syn', 'TFTP', 'DrDoS_NTP'],
    },
    'C': {
        'frame': modelC_train,
        'name': 'Model C',
        'description': "classifie les flux suivant les patternes d'attaques spécifiques",
        'classNumber': 17,
        'classes' : ['Portmap', 'NetBIOS', 'DrDoS_NetBIOS', 'DrDoS_DNS', 'DrDoS_LDAP', 'LDAP', 'DrDoS_SNMP', 'UDP', 'DrDoS_UDP', 'UDP-lag', 'Benign', 'DrDoS_MSSQL', 'MSSQL', 'WebDDoS', 'Syn', 'TFTP', 'DrDoS_NTP'],
    }
}
for modelFrame in modelsInfo.values():
    tk.Label(modelFrame['frame'], text=modelFrame['name'], fg=text_color_100, bg=bg_color_500, font=bigText).pack(pady=(20, 10), padx=10, fill='x')
    tk.Label(modelFrame['frame'], text=modelFrame['description'], fg=text_color_500, bg=bg_color_500, font=smallText, wraplength=300).pack(pady=(0, 10), padx=10, fill='x')
    tk.Label(modelFrame['frame'], text=f"Nombre de classes: {modelFrame['classNumber']}", fg=text_color_100, bg=bg_color_500, font=normalText_bold, justify='left').pack(pady=(10, 0), padx=10, fill='x')
    tk.Label(modelFrame['frame'], text=f"{', '.join(modelFrame['classes'])}", fg=text_color_500, bg=bg_color_500, font=verysmallText, height=5, justify='center', wraplength=300).pack(pady=(5, 10), padx=10, fill='x')
    tk.Button(modelFrame['frame'], text='Entrainer', font=normalText, bg=bg_color_500, fg=text_color_100, relief='groove', pady=5).pack(pady=(20, 10), padx=10, fill='x')







# test section
mainContainer_test = tk.Frame(mainContainer, bg=bg_color_500)
mainContainer_test.place(relx=0, rely=0, relheight=1, relwidth=1)
tk.Label(mainContainer_test, text='Evaluation des models', fg=text_color_100, bg=bg_color_500, font=bigText).pack(padx=10, pady=40, fill='x')

# Select Random Sample Frame
randomSampleFrame = tk.Frame(mainContainer_test, bg=bg_color_500)
randomSampleFrame.pack(padx=10, pady=(40,10))
GetRandomSampleBtn = tk.Button(randomSampleFrame, text='Nouvel échantillon', font=normalText, bg=bg_color_500, fg=text_color_100, relief='groove', command=selectRandomSample)
GetRandomSampleBtn.pack(pady=(0, 10), padx=(10, 100), side='left')
RandomSampleIndexLabel = tk.Label(randomSampleFrame, text="Échantillon 4512", fg=text_color_500, bg=bg_color_500, font=normalText_bold)
RandomSampleIndexLabel.pack(pady=(0, 10), padx=10, side='left')
RandomSampleClassLabel = tk.Label(randomSampleFrame, text="DRDoS_UDP", fg=text_color_100, bg=bg_color_500, font=normalText_bold, highlightcolor=bg_color_200, highlightbackground=bg_color_200, highlightthickness=2, padx=10, pady=5)
RandomSampleClassLabel.pack(pady=(0, 10), padx=10, side='left')

# testing result Frame
ResultShowcaseFrame = tk.Frame(mainContainer_test, bg=bg_color_500)
ResultShowcaseFrame.pack(padx=50, pady=(10,30), expand=False, fill='x')
ResultShowcaseFrame.grid_columnconfigure(1, weight=1)
ResultShowcaseFrame.grid_columnconfigure(2, weight=1)
ResultShowcaseFrame.grid_columnconfigure(3, weight=1)
# ResultShowcaseFrame.grid_rowconfigure(1, weight=1)
# ResultShowcaseFrame.grid_rowconfigure(2, weight=1)
# ResultShowcaseFrame.grid_rowconfigure(3, weight=1)
# title row
tk.Label(ResultShowcaseFrame, text='☺', fg=text_color_100, bg=bg_color_500, font=bigText).grid(row=0, column=0, padx=(10, 5), pady=(10, 0), sticky='nsew')
tk.Label(ResultShowcaseFrame, text='AIRS', fg=text_color_100, bg=bg_color_500, font=normalText_bold).grid(row=0, column=1, padx=(5, 5), pady=(10, 0), sticky='nsew')
tk.Label(ResultShowcaseFrame, text='RandomForst', fg=text_color_100, bg=bg_color_500, font=normalText_bold).grid(row=0, column=2, padx=(5, 5), pady=(10, 0), sticky='nsew')
tk.Label(ResultShowcaseFrame, text='Hybrid', fg=text_color_100, bg=bg_color_500, font=normalText_bold).grid(row=0, column=3, padx=(5, 5), pady=(10, 0), sticky='nsew')
# model column
tk.Label(ResultShowcaseFrame, text='Model A', fg=text_color_100, bg=bg_color_500, font=normalText_bold).grid(row=1, column=0, padx=(10, 15), pady=(10, 0), sticky='nsew')
tk.Label(ResultShowcaseFrame, text='Model B', fg=text_color_100, bg=bg_color_500, font=normalText_bold).grid(row=2, column=0, padx=(10, 15), pady=(10, 0), sticky='nsew')
tk.Label(ResultShowcaseFrame, text='Model C', fg=text_color_100, bg=bg_color_500, font=normalText_bold).grid(row=3, column=0, padx=(10, 15), pady=(10, 0), sticky='nsew')

# model A results
A_AIRS = tk.Label(ResultShowcaseFrame, text='Benign', fg=text_color_100, bg=good_color, font=normalText, relief='groove', pady=20)
A_AIRS.grid(row=1, column=1, padx=(5, 5), pady=(10, 0), sticky='nsew')
B_AIRS = tk.Label(ResultShowcaseFrame, text='Attack', fg=text_color_100, bg=good_color, font=normalText, relief='groove', pady=20)
B_AIRS.grid(row=2, column=1, padx=(5, 5), pady=(10, 0), sticky='nsew')
C_AIRS = tk.Label(ResultShowcaseFrame, text='Benign', fg=text_color_100, bg=good_color, font=normalText, relief='groove', pady=20)
C_AIRS.grid(row=3, column=1, padx=(5, 5), pady=(10, 0), sticky='nsew')
# model B results
A_RF = tk.Label(ResultShowcaseFrame, text='Portmap', fg=text_color_100, bg=good_color, font=normalText, relief='groove', pady=20)
A_RF.grid(row=1, column=2, padx=(5, 5), pady=(10, 0), sticky='nsew')
B_RF = tk.Label(ResultShowcaseFrame, text='Benign', fg=text_color_100, bg=good_color, font=normalText, relief='groove', pady=20)
B_RF.grid(row=2, column=2, padx=(5, 5), pady=(10, 0), sticky='nsew')
C_RF = tk.Label(ResultShowcaseFrame, text='DrDoS_NTP', fg=text_color_100, bg=bad_color, font=normalText, relief='groove', pady=20)
C_RF.grid(row=3, column=2, padx=(5, 5), pady=(10, 0), sticky='nsew')
# model C results
A_HYB = tk.Label(ResultShowcaseFrame, text='DrDoS_NetBIOS', fg=text_color_100, bg=bad_color, font=normalText, relief='groove', pady=20)
A_HYB.grid(row=1, column=3, padx=(5, 10), pady=(10, 0), sticky='nsew')
B_HYB = tk.Label(ResultShowcaseFrame, text='Benign', fg=text_color_100, bg=bad_color, font=normalText, relief='groove', pady=20)
B_HYB.grid(row=2, column=3, padx=(5, 10), pady=(10, 0), sticky='nsew')
C_HYB = tk.Label(ResultShowcaseFrame, text='DrDoS_NTP', fg=text_color_100, bg=bad_color, font=normalText, relief='groove', pady=20)
C_HYB.grid(row=3, column=3, padx=(5, 10), pady=(10, 0), sticky='nsew')

tk.Button(mainContainer_test, text='Nouvelle prédiction', font=normalText, bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=10, command=predictResults).pack(pady=(20, 30), padx=10)

selectRandomSample()   # to fill the initial table







# stat section
mainContainer_stat = tk.Frame(mainContainer, bg=bg_color_500)
mainContainer_stat.place(relx=0, rely=0, relheight=1, relwidth=1)
tk.Label(mainContainer_stat, text='Comparaison des models', fg=text_color_100, bg=bg_color_500, font=bigText).pack(padx=10, pady=(40,5), fill='x')

mainMainContainer_stat = tk.Frame(mainContainer_stat, bg=bg_color_500)
mainMainContainer_stat.pack(padx=10, pady=(0, 30), fill='both', expand=True)

# figure display
# image = tk.PhotoImage(file='.\\figures\\comparaison_accuracy.png')
image = Image.open('./figures/comparaison_accuracy.png')
image = image.resize((600, 400))
tkimage = ImageTk.PhotoImage(image)
imageFrame = tk.Label(mainMainContainer_stat, text='', image=tkimage, fg=text_color_500, bg=bg_color_500)
imageFrame.pack(side='left', padx=(0, 10), pady=(0, 10), fill='both', expand=True)
# imageFrame.pack(padx=10, pady=(0, 10), fill='x')


# figure selection
figureSelectFrame = tk.Frame(mainMainContainer_stat, bg=bg_color_500)
figureSelectFrame.pack(side='right', padx=(10, 20), pady=(0, 10), fill='y')

categorySelectFrame = tk.Frame(figureSelectFrame, bg=bg_color_500)
confMatSelectFrame = tk.Frame(figureSelectFrame, bg=bg_color_500)
otherMetricsFrame = tk.Frame(figureSelectFrame, bg=bg_color_500)
tk.Frame(figureSelectFrame, bg=bg_color_500, height=60).pack()  # spacer
categorySelectFrame.pack(pady=(0,40))
tk.Label(figureSelectFrame, text='Matrice de confusion', font=normalText_bold, bg=bg_color_500, fg=text_color_100).pack(pady=(20, 0))
confMatSelectFrame.pack(pady=(0,40))
# tk.Label(figureSelectFrame, text='Precision', font=normalText_bold, bg=bg_color_500, fg=text_color_100).pack(pady=(20, 0))
otherMetricsFrame.pack(pady=(0,40))



# tk.Label(confMatSelectFrame, text='metrics générales', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, padx=15, pady=0, justify='left').grid(row=0, column=0, sticky='ew')
tk.Button(categorySelectFrame, text='Accuracy', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('all', 'accuracy', '')).grid(row=0, column=1, sticky='ew')
tk.Button(categorySelectFrame, text='Precision', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('all', 'precision', '')).grid(row=0, column=2, sticky='ew')
tk.Button(categorySelectFrame, text='Recall', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('all', 'recall', '')).grid(row=1, column=1, sticky='ew')
tk.Button(categorySelectFrame, text='F1 Score', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('all', 'f1-score', '')).grid(row=1, column=2, sticky='ew')

DisplayFigure('all', 'accuracy', '')

# Confusion Matrix Selection
tk.Label(confMatSelectFrame, text='A', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, padx=15, pady=0, justify='left').grid(row=1, column=1, sticky='ew')
tk.Label(confMatSelectFrame, text='AIRS', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, padx=15, pady=0, justify='right').grid(row=2, column=0, sticky='ew')
tk.Button(confMatSelectFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('A', 'AIRS', 'confusion_matrix')).grid(row=2, column=1, sticky='ew')
tk.Button(confMatSelectFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('B', 'AIRS', 'confusion_matrix')).grid(row=2, column=2, sticky='ew')
tk.Button(confMatSelectFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('C', 'AIRS', 'confusion_matrix')).grid(row=2, column=3, sticky='ew')

tk.Label(confMatSelectFrame, text='B', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, padx=15, pady=0, justify='left').grid(row=1, column=2, sticky='ew')
tk.Label(confMatSelectFrame, text='RF', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, padx=15, pady=0, justify='right').grid(row=3, column=0, sticky='ew')
tk.Button(confMatSelectFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('A', 'SECOND', 'confusion_matrix')).grid(row=3, column=1, sticky='ew')
tk.Button(confMatSelectFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('B', 'SECOND', 'confusion_matrix')).grid(row=3, column=2, sticky='ew')
tk.Button(confMatSelectFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('C', 'SECOND', 'confusion_matrix')).grid(row=3, column=3, sticky='ew')

tk.Label(confMatSelectFrame, text='C', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, padx=15, pady=0, justify='left').grid(row=1, column=3, sticky='ew')
tk.Label(confMatSelectFrame, text='Hybrid', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, padx=15, pady=0, justify='right').grid(row=4, column=0, sticky='ew')
tk.Button(confMatSelectFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('A', 'HYB', 'confusion_matrix')).grid(row=4, column=1, sticky='ew')
tk.Button(confMatSelectFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('B', 'HYB', 'confusion_matrix')).grid(row=4, column=2, sticky='ew')
tk.Button(confMatSelectFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('C', 'HYB', 'confusion_matrix')).grid(row=4, column=3, sticky='ew')

metricToShow = tk.StringVar(value='precision')  # default metric

def otherMetricsSelect(selectedMetric):
    if selectedMetric == 'precision':
        precisionBtn.configure(relief='groove', bg=bg_color_500, fg=text_color_100)
        recallBtn.configure(relief='flat', bg=bg_color_400, fg=text_color_500)
        f1Btn.configure(relief='flat', bg=bg_color_400, fg=text_color_500)

        metricToShow.set('precision')

    elif selectedMetric == 'recall':
        precisionBtn.configure(relief='flat', bg=bg_color_400, fg=text_color_500)
        recallBtn.configure(relief='groove', bg=bg_color_500, fg=text_color_100)
        f1Btn.configure(relief='flat', bg=bg_color_400, fg=text_color_500)

        metricToShow.set('recall')

    elif selectedMetric == 'f1-score':
        precisionBtn.configure(relief='flat', bg=bg_color_400, fg=text_color_500)
        recallBtn.configure(relief='flat', bg=bg_color_400, fg=text_color_500)
        f1Btn.configure(relief='groove', bg=bg_color_500, fg=text_color_100)

        metricToShow.set('f1-score')



# other metrics Matrix Selection
otherMetricsBtns = tk.Frame(otherMetricsFrame, bg=bg_color_500)
otherMetricsBtns.grid(row=0, column=0, columnspan=5, sticky='ew', pady=(10, 0))

precisionBtn = tk.Button(otherMetricsBtns, text='Precision', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : otherMetricsSelect('precision'))
recallBtn = tk.Button(otherMetricsBtns, text='Recall', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : otherMetricsSelect('recall'))
f1Btn = tk.Button(otherMetricsBtns, text='f1-score', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : otherMetricsSelect('f1-score'))
precisionBtn.grid(row=0, column=0, sticky='ew')
recallBtn.grid(row=0, column=1, sticky='ew')
f1Btn.grid(row=1, column=0, columnspan=2, sticky='ew')



tk.Label(otherMetricsFrame, text='A', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, padx=15, pady=0, justify='left').grid(row=1, column=1, sticky='ew')
tk.Label(otherMetricsFrame, text='AIRS', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, padx=15, pady=0, justify='right').grid(row=2, column=0, sticky='ew')
tk.Button(otherMetricsFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('A', 'AIRS', metricToShow.get())).grid(row=2, column=1, sticky='ew')
tk.Button(otherMetricsFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('B', 'AIRS', metricToShow.get())).grid(row=2, column=2, sticky='ew')
tk.Button(otherMetricsFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('C', 'AIRS', metricToShow.get())).grid(row=2, column=3, sticky='ew')

tk.Label(otherMetricsFrame, text='B', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, padx=15, pady=0, justify='left').grid(row=1, column=2, sticky='ew')
tk.Label(otherMetricsFrame, text='RF', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, padx=15, pady=0, justify='right').grid(row=3, column=0, sticky='ew')
tk.Button(otherMetricsFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('A', 'SECOND', metricToShow.get())).grid(row=3, column=1, sticky='ew')
tk.Button(otherMetricsFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('B', 'SECOND', metricToShow.get())).grid(row=3, column=2, sticky='ew')
tk.Button(otherMetricsFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('C', 'SECOND', metricToShow.get())).grid(row=3, column=3, sticky='ew')

tk.Label(otherMetricsFrame, text='C', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, padx=15, pady=0, justify='left').grid(row=1, column=3, sticky='ew')
tk.Label(otherMetricsFrame, text='Hybrid', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, padx=15, pady=0, justify='right').grid(row=4, column=0, sticky='ew')
tk.Button(otherMetricsFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('A', 'HYB', metricToShow.get())).grid(row=4, column=1, sticky='ew')
tk.Button(otherMetricsFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('B', 'HYB', metricToShow.get())).grid(row=4, column=2, sticky='ew')
tk.Button(otherMetricsFrame, text='  ', font=('Helvetica', 12), bg=bg_color_500, fg=text_color_100, relief='groove', padx=15, pady=0, command=lambda : DisplayFigure('C', 'HYB', metricToShow.get())).grid(row=4, column=3, sticky='ew')









root.mainloop()



