import PySimpleGUI as sg
import os.path

# First the window layout in 2 columns

file_list_column = [
    [
        sg.Text("Welcome to Speaker Recognition Tool!\n"),
        sg.Text("1) choose file or choose device to record:\n"),
    ],
    [
        sg.Text("File to analyse"),
        sg.In(size=(25, 1), enable_events=True, key="-FILENAME-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Text("Device to record"),
        #список устройств с возможностью выбрать
    ],
    [
        sg.Text("2) choose identification or verification mode (for verification don't forget to enter speaker ID)\n"),
    ],
            # галочки () идентификация () верификация (+ поле для ввода идентификатора говорящего)
    [
        sg.Button("Analyse", key="-ANALYSE-"),
    ],
    [   
        sg.Text("Speakers list:\n"),
        sg.Listbox(
            #выводить существующие идентификаторы спикеров
            values=[], enable_events=True, size=(40, 10), key="-FILE LIST-"
        ),
    ],
    [
        sg.Text("If you have new checkpoint or there were some changes in base, you should update the base:\n"),
        sg.Button("Update base", key="-UPDATE-"),
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]


window = sg.Window("Speaker Recognition Tool", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FILENAME-":
        folder = values["-FILENAME-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FILENAME-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(filename=filename)

        except:
            pass

window.close()
