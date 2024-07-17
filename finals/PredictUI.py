import os

from PIL._tkinter_finder import tk

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pathlib import Path
from tkinter import Tk, Canvas, Button, Entry, Checkbutton, IntVar, Toplevel, Label, Frame, BOTH, Scrollbar
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

model_path = r"..\res\lstm_model.keras"
model = load_model(model_path)

OUTPUT_PATH = Path(__file__).parent
IMAGE_PATH = Path(r"..\res\lol-logo.png")

# Define the columns to be used for input
all_columns = [
    'gamelength', 'k', 'd', 'a', 'teamkills', 'teamdeaths', 'doubles', 'triples',
    'quadras', 'pentas', 'fb', 'fbassist', 'fbvictim', 'fbtime', 'kpm', 'okpm',
    'ckpm', 'fd', 'fdtime', 'teamdragkills', 'oppdragkills', 'elementals',
    'oppelementals', 'firedrakes', 'waterdrakes', 'earthdrakes', 'airdrakes',
    'elders', 'oppelders', 'herald', 'heraldtime', 'ft', 'fttime',
    'firstmidouter', 'firsttothreetowers', 'teamtowerkills', 'opptowerkills',
    'fbaron', 'fbarontime', 'teambaronkills', 'oppbaronkills', 'dmgtochamps',
    'dmgtochampsperminute', 'dmgshare', 'earnedgoldshare', 'wards', 'wpm',
    'wardshare', 'wardkills', 'wcpm', 'visionwards', 'visionwardbuys',
    'visiblewardclearrate', 'invisiblewardclearrate', 'totalgold', 'earnedgpm',
    'goldspent', 'gspd', 'minionkills', 'monsterkills', 'monsterkillsownjungle',
    'monsterkillsenemyjungle', 'cspm', 'goldat10', 'oppgoldat10', 'gdat10',
    'goldat15', 'oppgoldat15', 'gdat15', 'xpat10', 'oppxpat10', 'xpdat10',
    'csat10', 'oppcsat10', 'csdat10', 'csat15', 'oppcsat15', 'csdat15'
]

def relative_to_assets(path: str) -> Path:
    return OUTPUT_PATH / Path(path)


def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")


def open_predict_ui():
    top = Toplevel(window)
    top.title("Select Columns")
    center_window(top, 500, 500)

    selected_columns = [IntVar() for _ in all_columns]

    # Create a frame for the Checkbuttons and a scrollbar
    frame = Frame(top)
    frame.pack(fill=BOTH, expand=True)

    canvas = Canvas(frame)
    scroll_y = Scrollbar(frame, orient="vertical", command=canvas.yview)
    scroll_y.pack(side="right", fill="y")

    canvas.pack(side="left", fill=BOTH, expand=True)
    canvas.configure(yscrollcommand=scroll_y.set)

    # Create a frame inside the canvas to hold the Checkbuttons
    checkbox_frame = Frame(canvas)
    canvas.create_window((0, 0), window=checkbox_frame, anchor="nw")

    for i, column in enumerate(all_columns):
        chk = Checkbutton(checkbox_frame, text=column, variable=selected_columns[i])
        chk.pack(anchor='w')

    # Update the scrollbar region
    checkbox_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def apply_selection():
        selected = [all_columns[i] for i in range(len(all_columns)) if selected_columns[i].get() == 1]
        top.destroy()
        show_prediction_ui(selected)

    apply_button = Button(top, text="Apply", command=apply_selection, width=10, height=5, font=("Arial", 8))
    apply_button.pack(pady=8)


def show_prediction_ui(selected_columns):
    top = Toplevel(window)
    top.title("Prediction Input")
    center_window(top, 1200, 700)

    # Create a frame for the table layout
    frame = Frame(top)
    frame.pack(fill=BOTH, expand=True)

    # Create headers for the table
    Label(frame, text="Player").grid(row=0, column=0)
    Label(frame, text="Team").grid(row=0, column=1)
    Label(frame, text="Position").grid(row=0, column=2)
    Label(frame, text="Champion").grid(row=0, column=3)
    for ban_col in range(5):  # Labels for 5 bans
        Label(frame, text=f"Ban {ban_col + 1}").grid(row=0, column=4 + ban_col)
    for i, column in enumerate(selected_columns):
        Label(frame, text=column).grid(row=0, column=i + 9)

    # Retrieve team names and player positions from main window
    blue_team_name = blue_team_entry.get()
    red_team_name = red_team_entry.get()
    blue_player_positions = [role for role, entry in zip(player_roles, blue_players) if entry.get()]
    red_player_positions = [role for role, entry in zip(player_roles, red_players) if entry.get()]

    # Create entry boxes for each player and their inputs
    player_entries = []
    ban_entries = []
    for i in range(5):  # 5 players per team
        blue_player_number = i + 1
        red_player_number = i + 6  # Starting from row 6 for Red Team

        # Blue team entries
        blue_player_entry = Entry(frame, width=15)
        blue_player_entry.grid(row=blue_player_number, column=0)
        blue_player_entry.insert(0, f"Blue Player {blue_player_number}")
        player_entries.append(blue_player_entry)

        Label(frame, text=blue_team_name).grid(row=blue_player_number, column=1)

        position_entry_blue = Entry(frame, width=15)
        position_entry_blue.grid(row=blue_player_number, column=2)
        position_entry_blue.insert(i, blue_player_positions[i])

        # Champion entry
        champion1_entry_blue = Entry(frame, width=15)
        champion1_entry_blue.grid(row=1, column=3)
        champion2_entry_blue = Entry(frame, width=15)
        champion2_entry_blue.grid(row=2, column=3)
        champion3_entry_blue = Entry(frame, width=15)
        champion3_entry_blue.grid(row=3, column=3)
        champion4_entry_blue = Entry(frame, width=15)
        champion4_entry_blue.grid(row=4, column=3)
        champion5_entry_blue = Entry(frame, width=15)
        champion5_entry_blue.grid(row=5, column=3)
        blue_champions = [champion1_entry_blue, champion2_entry_blue, champion3_entry_blue,
                          champion4_entry_blue, champion5_entry_blue]

        # Bans for Blue Team
        blue_ban_entries = []
        for ban_col in range(5):
            ban_entry_blue = Entry(frame, width=15)
            ban_entry_blue.grid(row=blue_player_number, column=4 + ban_col)
            blue_ban_entries.append(ban_entry_blue)
        ban_entries.append(blue_ban_entries)

        # Additional entries based on selected columns
        col_entries_blue = []
        for j, col in enumerate(selected_columns):
            entry_blue = Entry(frame, width=15)
            entry_blue.grid(row=blue_player_number, column=j + 9)
            col_entries_blue.append(entry_blue)

        # Red team entries
        red_player_entry = Entry(frame, width=15)
        red_player_entry.grid(row=red_player_number, column=0)
        red_player_entry.insert(0, f"Red Player {i + 1}")
        player_entries.append(red_player_entry)

        Label(frame, text=red_team_name).grid(row=red_player_number, column=1)

        position_entry_red = Entry(frame, width=15)
        position_entry_red.grid(row=red_player_number, column=2)
        position_entry_red.insert(i, red_player_positions[i])

        # Champion entry
        champion1_entry_red = Entry(frame, width=15)
        champion1_entry_red.grid(row=6, column=3)
        champion2_entry_red = Entry(frame, width=15)
        champion2_entry_red.grid(row=7, column=3)
        champion3_entry_red = Entry(frame, width=15)
        champion3_entry_red.grid(row=8, column=3)
        champion4_entry_red = Entry(frame, width=15)
        champion4_entry_red.grid(row=9, column=3)
        champion5_entry_red = Entry(frame, width=15)
        champion5_entry_red.grid(row=10, column=3)
        red_champions = [champion1_entry_red, champion2_entry_red, champion3_entry_red,
                          champion4_entry_red, champion5_entry_red]

        # Bans for Red Team
        red_ban_entries = []
        for ban_col in range(5):
            ban_entry_red = Entry(frame, width=15)
            ban_entry_red.grid(row=red_player_number, column=4 + ban_col)
            red_ban_entries.append(ban_entry_red)
        ban_entries.append(red_ban_entries)

        # Additional entries based on selected columns
        col_entries_red = []
        for j, col in enumerate(selected_columns):
            entry_red = Entry(frame, width=15)
            entry_red.grid(row=red_player_number, column=j + 9)
            col_entries_red.append(entry_red)

    def predict():
        # Create a DataFrame to store input data
        input_data = []
        for i in range(5):  # 5 players per team
            blue_player_data = {
                'player': player_entries[(i * 2)].get(),
                'team': blue_team_name,
                'position': blue_player_positions[i],
                'champion': blue_champions[i].get(),
                'side': 'Blue',
            }
            for ban_col in range(5):  # Only iterate over 5 bans
                blue_player_data[f'ban{ban_col+1}'] = ban_entries[0][ban_col].get()
            for j, col in enumerate(selected_columns):
                blue_player_data[col] = col_entries_blue[j].get()
            input_data.append(blue_player_data)

            red_player_data = {
                'player': player_entries[(i * 2) + 1].get(),
                'team': red_team_name,
                'position': red_player_positions[i],
                'champion': red_champions[i].get(),
                'side': 'Red',
            }
            for ban_col in range(5):  # Only iterate over 5 bans
                red_player_data[f'ban{ban_col+1}'] = ban_entries[1][ban_col].get()
            for j, col in enumerate(selected_columns):
                red_player_data[col] = col_entries_red[j].get()
            input_data.append(red_player_data)

        input_df = pd.DataFrame(input_data)

        # Ensure columns are numeric where needed
        numeric_columns = input_df.columns.difference(['side', 'player', 'team', 'position', 'champion', 'ban1', 'ban2',
                                                       'ban3', 'ban4', 'ban5'])
        input_df[numeric_columns] = input_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        input_df = input_df.fillna(0)  # Replace NaNs with 0 for numeric columns

        predict_and_display(input_df)

    # Button to predict
    Button(top, text="Predict", command=predict).pack(pady=10)


def predict_and_display(input_df):
    # Encoding categorical variables
    label_encoders = {}
    categorical_columns = ['side', 'position', 'player', 'team', 'champion', 'ban1', 'ban2', 'ban3', 'ban4', 'ban5']
    for column in categorical_columns:
        le = LabelEncoder()
        input_df[column] = le.fit_transform(input_df[column].astype(str))
        label_encoders[column] = le

    # Scaling features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(input_df)

    # Check and adjust the number of features to match the model's input shape
    expected_num_features = model.input_shape[2]
    current_num_features = features_scaled.shape[1]

    if current_num_features > expected_num_features:
        features_scaled = features_scaled[:, :expected_num_features]
    elif current_num_features < expected_num_features:
        # Add zero columns if there are fewer features than expected
        features_scaled = np.pad(features_scaled, ((0, 0), (0, expected_num_features - current_num_features)),
                                 'constant')

    # Reshape for LSTM [samples, time steps, features]
    features_scaled = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))

    # Make predictions
    predictions = model.predict(features_scaled)
    predicted_results = (predictions > 0.5).astype(int).flatten()

    result_window = Toplevel(window)
    result_window.title("Prediction Results")
    center_window(result_window, 300, 200)

    Label(result_window, text=f"Predicted Result: {'Win' if predicted_results[0] == 1 else 'Loss'}").pack()


window = Tk()
window.title("League of Legends Predictor")
window.configure(bg="#1B2641")

canvas = Canvas(window, bg="#1B2641", height=500, width=1000, bd=0, highlightthickness=0, relief="ridge")
canvas.place(x=0, y=0)

canvas.create_rectangle(0.0, 0.0, 255.0, 500.0, fill="#091428", outline="")

predict_button = Button(window, text="Predict", command=open_predict_ui, font=("Jockey One", 20), bg="#091428",
                        fg="#C89B3C", bd=0, highlightthickness=0)
predict_button.place(x=80, y=265, anchor="nw")

# Load and resize the logo image
original_image = Image.open(IMAGE_PATH)
resized_image = original_image.resize((100, 100), Image.LANCZOS)
logo_image = ImageTk.PhotoImage(resized_image)
canvas.create_image(127, 87, image=logo_image)

canvas.create_text(350.0, 30.0, anchor="nw", text="BLUE TEAM", fill="#FFFFFF", font=("Jockey One", 32))
canvas.create_text(750.0, 30.0, anchor="nw", text="RED TEAM", fill="#FFFFFF", font=("Jockey One", 32))

blue_team_entry = Entry(window, bd=0, bg="#D9D9D9", highlightthickness=0)
blue_team_entry.place(x=384.0, y=100.0, width=204.0, height=22.0)

red_team_entry = Entry(window, bd=0, bg="#D9D9D9", highlightthickness=0)
red_team_entry.place(x=776.0, y=100.0, width=204.0, height=22.0)

canvas.create_text(292.0, 100.0, anchor="nw", text="Team Name:", fill="#FFFFFF", font=("Jockey One", 12))
canvas.create_text(684.0, 100.0, anchor="nw", text="Team Name:", fill="#FFFFFF", font=("Jockey One", 12))

player_roles = ['Top', 'Jungle', 'Mid', 'ADC', 'Support']

blue_players = []
red_players = []
red_champions = []
blue_champions = []

for i, role in enumerate(player_roles):
    Label(window, text=f"{role}:", bg="#1B2641", fg="#FFFFFF", font=("Jockey One", 12)).place(x=292, y=176 + (i * 30))
    Label(window, text=f"{role}:", bg="#1B2641", fg="#FFFFFF", font=("Jockey One", 12)).place(x=684, y=176 + (i * 30))

    blue_player_entry = Entry(window, bd=0, bg="#D9D9D9", highlightthickness=0)
    blue_player_entry.place(x=384.0, y=176.0 + (i * 30), width=204.0, height=22.0)
    blue_players.append(blue_player_entry)

    red_player_entry = Entry(window, bd=0, bg="#D9D9D9", highlightthickness=0)
    red_player_entry.place(x=776.0, y=176.0 + (i * 30), width=204.0, height=22.0)
    red_players.append(red_player_entry)

window.resizable(False, False)
center_window(window, 1000, 500)
window.mainloop()
