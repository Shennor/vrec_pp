import PySimpleGUI as sg
from check import *
from conv_models import DeepSpeakerModel
import os.path

sg.theme("NeutralBlue")

ids = get_id_dict()
#print('1')
#print(ids)
names = list(ids.values())
#print(names)
devices = get_device_list()

file_list_column = [
        [sg.Text("Welcome to Speaker Recognition Tool!")],
        [sg.Text("1) choose file or choose device to record, to record from default output click on button below:")],
        [
            sg.Text("File to analyse"),
            sg.In(size=(25, 1), enable_events=True, key="-FILENAME-"),
            sg.FileBrowse(),
        ],
        [
            sg.Text("Devices:"),
            sg.Listbox(values=devices, enable_events=True, size=(40, 8), key="-DEVICE LIST-"),
        #список устройств с возможностью выбрать
        ],
        [sg.Checkbox("Record from default output", default=False, enable_events=True, key="-DEFAULT OUTPUT-")],
        [sg.Text('_'*80)],
        [sg.Text("2) choose speaker from list (for verification or for editting his name")],
        [sg.Text("Speakers list:")],
        [
            sg.Listbox(values=names, enable_events=True, size=(40, 10), key="-NAMES LIST-"),
        ],
        [
            sg.Text("New name: "),
            sg.In(size=(25, 1), enable_events=True, key="-NEW NAME-"),
        ],
        [sg.Button('Rename', key="-RENAME-")],
        [sg.Text('_'*80)],
        [sg.Text("3) choose identification or verification mode")],
        [   
            sg.Button('Identify', key="-IDENTIFY-"),
            sg.Button('Verify', key="-VERIFY-"),
        ],
        [sg.Text('_'*80)],
        [sg.Text("If you have new checkpoint or there were some changes in base, you should update the base"),
        sg.Button("Update base", key="-UPDATE-"),
        ],
]

result_viewer_column = [
    [sg.Text("Output:")],
    [sg.Multiline(size=(50, 40), key="-OUTPUT-")],
    [sg.Button("Clear", key="-CLEAR-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(result_viewer_column),
    ]
]


window = sg.Window("Speaker Recognition Tool", layout)

# Run model
model = DeepSpeakerModel()
checkpoint = 'ResCNN_checkpoint_36.h5'
model.m.load_weights(checkpoint, by_name=True)

filename = None
speaker_id = None
device_id = None
id_dict = get_id_dict()
new_name = None
old_name = None
default = False


sg.cprint_set_output_destination(window, "-OUTPUT-")

while True:
    event, values = window.read() 
    #sg.cprint_set_output_destination(window, "-OUTPUT-")
    #default = window["-DEFAULT OUTPUT-"].get()
    #sg.cprint(default)
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    elif event == "-FILENAME-":
        filename = values["-FILENAME-"]
        device_id = None
    elif event == "-DEVICE LIST-":
        device_id = int(values["-DEVICE LIST-"][0].split(' ')[0])
        filename = None
   # elif event == "-DEFAULT OUTPUT-":
    #    default = values["-DEFAULT OUTPUT-"]       
    elif event == "-NAMES LIST-":
        old_name = values["-NAMES LIST-"][0]
        speaker_id = id_from_name(old_name, ids)
    elif event == "-RENAME-":
        if old_name == None:
            sg.cprint("Please choose speaker from list to rename him/her...")
        elif new_name == None or new_name.replace(' ', '') == '':
            sg.cprint("Please enter new speaker name to rename him/her...")
        else:
            rename_student(old_name, values["-NEW NAME-"], ids)
        window["-NAMES LIST-"].update(names_list())
    elif event == "-NEW NAME-":
        new_name = values["-NEW NAME-"]
    elif event == "-IDENTIFY-" or event == "-VERIFY-":
        pred_tensor = []
        #sg.cprint('1')
        if values["-DEFAULT OUTPUT-"]:
            pred_tensor = predict_default(model)
            #sg.cprint('4')
        elif not filename == None:
            pred_tensor = predict_by_file(filename, model)
            #sg.cprint('2')
        elif not device_id == None:
            #sg.cprint("Say something...")
            pred_tensor = predict_by_id(device_id, model)
            #sg.cprint("Record finished!")
            #sg.cprint('3')
        else:
            sg.cprint("Please choose file or device to record or click 'Record from default output'!")
        if not pred_tensor == []:
            if event == "-IDENTIFY-":
                base = find_statistics(pred_tensor, PREDICTED_BASE)
                print_statistics(base)
            else:
                if not speaker_id == None:
                    base = verify_student(pred_tensor, speaker_id)
                    print_verification_result(base)
                else:
                    sg.cprint("You've forgot to choose id from list!")

    elif event == "-UPDATE-":
        sg.cprint("Running the model and making prediction for all available samples. Please, wait a few minutes...")
        make_student_prediction(model)
        new_ids = id_list()
        window["-ID LIST-"].update(new_ids)
    elif event == "-CLEAR-":
        window["-OUTPUT-"].update("")

window.close()
