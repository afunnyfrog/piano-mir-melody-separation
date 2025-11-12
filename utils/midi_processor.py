from mido import MidiFile, MidiTrack, merge_tracks
import os

# 這行程式碼會取得「目前這支 .py 檔案」所在的「絕對路徑」
# 例如：'f:\專題\piano-mir-melody-separation\utils'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 假設你的 MIDI 檔案名稱
midi_filename = 'on_berg-alban-piano-sonata-opus-1.mid'

# 接著，用 os.path.join() 把它們組合成一個「絕對不會錯」的路徑
MIDI_FILE_PATH = os.path.join(SCRIPT_DIR, midi_filename)

def separate_channels(input_midi_path):
    """
    讀取一個 MIDI 檔案，並將其 Channel 0 和 Channel 1 的音符
    分離成兩個獨立的 MIDI 檔案。
    """
    
    # 載入來源 MIDI 檔案
    try:
        mixed_midi = MidiFile(input_midi_path)
    except Exception as e:
        print(f"error：無法讀取 MIDI 檔案 {input_midi_path}。{e}")
        return

    # 建立兩個新的 MIDI 檔案物件
    melody_midi = MidiFile(type=mixed_midi.type)
    accomp_midi = MidiFile(type=mixed_midi.type)

    # 建立兩個新的音軌
    melody_track = MidiTrack()
    accomp_track = MidiTrack()

    melody_midi.tracks.append(melody_track)
    accomp_midi.tracks.append(accomp_track)

    print(f"正在處理檔案：{input_midi_path}")

    # 遍歷來源檔案中的所有音軌 (通常鋼琴獨奏只有一個音軌)
    for track in mixed_midi.tracks:
        for msg in track:
            # 我們只關心 note_on 和 note_off 訊息
            if msg.type == 'note_on' or msg.type == 'note_off':
                
                # 檢查 MIDI 通道 (Channel)
                if msg.channel == 0:
                    # 複製訊息並加入到「旋律」音軌
                    melody_track.append(msg.copy())
                    
                elif msg.channel == 1:
                    # 複製訊息並加入到「伴奏」音軌
                    # 注意：我們把 channel 改回 0，因為它是新檔案中的第一個音軌
                    new_msg = msg.copy(channel=0) 
                    accomp_track.append(new_msg)
            
            # 你也可以複製其他非音符訊息 (如：節拍、調號)
            # 這裡為了簡潔，我們先專注在音符上

    # 產生輸出的檔案名稱
    output_melody_path = input_midi_path.replace('.mid', '_melody.mid')
    output_accomp_path = input_midi_path.replace('.mid', '_accompaniment.mid')

    # 儲存兩個新的 MIDI 檔案
    melody_midi.save(output_melody_path)
    print(f"旋律 成功儲存檔案 -> {output_melody_path}")
    
    accomp_midi.save(output_accomp_path)
    print(f"伴奏 成功儲存檔案 -> {output_accomp_path}")


# --- 這裏是主程式 ---
if __name__ == "__main__":
    # 假設你從 MuseScore 下載了一個檔案，叫做 "moonlight_sonata.mid"
    # 並且你已經驗證過它在播放軟體中是「兩種顏色」（使用 Channel 0 和 1）
    
    # 執行分離
    separate_channels(MIDI_FILE_PATH)

    # 你現在就會得到 "moonlight_sonata_melody.mid" 和 
    # "moonlight_sonata_accompaniment.mid" 兩個檔案