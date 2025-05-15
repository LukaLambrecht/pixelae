# define numbering of ROCs within a module
ROC_NUMBERS = {
      (0,0): 15,
      (0,1): 14,
      (0,2): 13,
      (0,3): 12,
      (0,4): 7,
      (0,5): 6,
      (0,6): 5,
      (0,7): 4,
      (1,0): 11,
      (1,1): 10,
      (1,2): 9,
      (1,3): 8,
      (1,4): 3,
      (1,5): 2,
      (1,6): 1,
      (1,7): 0
}

ROC_COORDS = {
      15: (0,0),
      14: (0,1),
      13: (0,2),
      12: (0,3),
      7: (0,4),
      6: (0,5),
      5: (0,6),
      4: (0,7),
      11: (1,0),
      10: (1,1),
      9: (1,2),
      8: (1,3),
      3: (1,4),
      2: (1,5),
      1: (1,6),
      0: (1,7)
}

def roc_to_coords(roc):
    '''
    Get the coordinates of a given ROC number inside a module.
    Note: the returned coordinates are relative to the module
          and in the usual array plotting convention,
          i.e. the first (y-) coordinate is 0 (upper row) or 1 (lower row),
          and the second (x-) coordinate can go from 0 (left) to 7 (right)
    '''
       
    # determine the coordinates for the given ROC
    res = ROC_COORDS[roc]
    return res


def coords_to_roc(coords):
    '''
    Get the ROC number from given coordinates.
    Note: the coordinates must be relative to the module
          and in the usual array plotting convention,
          i.e. the first (y-) coordinate must be 0 (upper row) or 1 (lower row),
          and the second (x-) coordinate can go from 0 (left) to 7 (right)
    '''
    
    # determine the ROC for the given coordinates
    res = ROC_NUMBERS[coords]
    return res