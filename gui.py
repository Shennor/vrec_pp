import PySimpleGUI as sg
from check import id_list, predict_by_file, predict_default, predict_by_id, verify_student, print_verification_result, find_statistics, print_statistics, PREDICTED_BASE, make_student_prediction, get_device_list
from conv_models import DeepSpeakerModel
import os.path

sg.theme("NeutralBlue")

ids = id_list()
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
        [sg.Radio("Record from default output", "RADIO1", key="-DEFAULT OUTPUT-")],
        [sg.Text('_'*80)],
        [sg.Text("2) choose identification or verification mode (for verification don't forget to enter speaker ID)")],
        [sg.Text("Speakers list:")],
        [
            sg.Listbox(values=ids, enable_events=True, size=(40, 10), key="-ID LIST-"),
        ],
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


while True:
    event, values = window.read() 
    sg.cprint_set_output_destination(window, "-OUTPUT-")
    default = window["-DEFAULT OUTPUT-"].get()
    sg.cprint(default)

    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    elif event == "-FILENAME-":
        filename = values["-FILENAME-"]
        device_id = None
    elif event == "-DEVICE LIST-":
        device_id = int(values["-DEVICE LIST-"][0].split(' ')[0])
        filename = None
    elif event == "-ID LIST-":
        speaker_id = values["-ID LIST-"][0]
    elif event == "-IDENTIFY-" or event == "-VERIFY-":
        pred_tensor = []
        sg.cprint('1')
        if default:
            pred_tensor = predict_default(model)
            sg.cprint('4')
        elif not filename == None:
            pred_tensor = predict_by_file(filename, model)
            sg.cprint('2')
        elif not device_id == None:
            #sg.cprint("Say something...")
            pred_tensor = predict_by_id(device_id, model)
            #sg.cprint("Record finished!")
            sg.cprint('3')
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
