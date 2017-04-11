import pandas as pd

labels = ['Boom', 'Stick', 'Bucket', 'Swing']


def preprocess_blended(blended):
    ''' Clips blended ctrl signals at  [-2, 2], and sets to zero where blending is off '''

    for lbl in labels:
        blended[lbl + ' Ctrl'] *= blended['Confidence']
        blended[lbl + ' Ctrl'] = blended[lbl + ' Ctrl'].clip(-1, 1, 0)

    return blended
