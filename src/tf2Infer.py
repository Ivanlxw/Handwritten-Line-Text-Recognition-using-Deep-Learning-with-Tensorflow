import os

from tf2Custom import *
from SpellChecker import correct_sentence

def infer(model, fnImg, correct):
    """ Recognize text in image provided by file path """
    img = preprocessor(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), imgSize=Model.imgSize)
    if img is None:
        print("Image not found")
        return

    recognized = model.predict(fnImg)
    print(recognized)
    recognized = model.decoderToText(recognized)

    print("Without Correction: ", recognized)
    print("With Correction: ", correct_sentence(recognized))
    if correct:
        return correct_sentence(recognized)
    else:    
        return recognized

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image to be OCR'd")
    args = vars(ap.parse_args())
    return args

if __name__ == "__main__":
    from DataLoader import FilePaths
    args = parse_args()
    model = Model(open(FilePaths.fnCharList).read(),
                      decoderType)
    infer_model = model.get_model()
    ## saved fp
    saved_fp = os.path.join(os.getcwd(), "../train_scratch")
    infer_model.load_weights("ckpt") 
    infer(infer_model, args["image"])