from models import *

def main():
    P = prompt_video("Did the last person open the bottle without using a knife?")
    vid = video_frame("sj81PWrerDk.mp4",64)
    inputs = process_inputs(P, vid)
    output_result = output(inputs)
    return output_result

if __name__ == "__main__":
    result = main()
    print(result)