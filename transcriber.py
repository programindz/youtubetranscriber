from moviepy.video.io.VideoFileClip import VideoFileClip
import gradio as gr
import whisper
import pytube
from openai import OpenAI


#############################################################################
# Using OpenAI API, need account credits for this
'''
client = OpenAI(api_key="Your API Key")
def transcribe_with_openai(audio_file):
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file = open(audio_file, 'rb')
    )
    return result["text"]
'''
#############################################################################

model = whisper.load_model("base", in_memory=True)

def transcribe_audio_with_whisper(audio_file):
    result = model.transcribe(audio_file)
    return result["text"]


def download_youtube_video(link):
    data = pytube.YouTube(link)
    audio = data.streams.get_by_itag(251)
    filename = audio.download()
    return filename

#########################################################
# Use In Case of Video To Audio Conversion

def video_to_audio(input_video):
    video_clip = VideoFileClip(input_video)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(f"{input_video}.wav")

    return f"{input_video}.wav"

##########################################################  
    
def transcribe_link(link):
    youtube_video_audio = download_youtube_video(link)
    return transcribe_audio_with_whisper(youtube_video_audio)
    


url_tab = gr.Interface(
    transcribe_link,
    inputs= gr.Text(label="Youtube Video URL"),
    outputs= "text"
)

audio_tab = gr.Interface(
    transcribe_audio_with_whisper,
    inputs = gr.Audio(type='filepath', label='Audio', sources=['upload', 'microphone'], format='mp3'),
    outputs= "text"
    
)

video_tab = gr.Interface(
    transcribe_audio_with_whisper,
    inputs = gr.Video(label="Video", sources=['upload','webcam']),
    outputs="text"
)



if __name__ == "__main__":
    app =  gr.TabbedInterface([url_tab, audio_tab, video_tab], ['URL','Audio', 'Video'])
    app.launch()
