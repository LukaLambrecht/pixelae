# Numbering of ROCs within a module

# From staring for a long time at a lot of examples, it seems ROCs are numbered as follows.
# For modules that are mounted on positive even (+2, +4, ...) or negative odd (-1, -3, ...) ladders:
#   | 7  6  5  4  3  2  1  0  |
#   | 8  9  10 11 12 13 14 15 |
# For modules that are mounted on positive odd (+1, +3, ...) or negative even (-2, -4, ...) ladders:
#   | 8  9  10 11 12 13 14 15 |
#   | 7  6  5  4  3  2  1  0  |

# For the numbers above, the orientation is as it appears in the monitoring elements.
# In a plotting convention where the origin is in the lower left corner of the monitoring element,
# this implies, for example for a module on a positive even ladder:
#   8  <-> (0, 0)
#   9  <-> (0, 1)
#   10 <-> (0, 2)
#   ...
#   7  <-> (1, 0)
#   6  <-> (1, 1)
#   ...

# I didn't find any documentation on this, so it's just a guess based on some (many) examples.
# Although groups of 4 neighbouring ROCs are probably fine, I'm not perfectly sure about the numbering
# of individual ROCs within these groups, as automasking of individual ROCs does not happen very often.

# Update: all of the above was for BPix 1. The plot seems to thicken however for other layers...
# BPix 2: the above is correct but only for positive module numbers. For negative module numbers, seems to be reversed...
# BPix 3: seems to be the same as BPix 2.
# BPix 4: seems to be the same as BPix 2.

# Todo: check also for other layers than layer 1.


def get_schema(layer, ladder, module):
    if layer == 1:
        if( (ladder>0 and ladder%2==0) or (ladder<0 and ladder%2==1) ): return 0
        else: return 1
    if layer == 2 or layer == 3 or layer == 4:
        if module > 0:
            if( (ladder>0 and ladder%2==0) or (ladder<0 and ladder%2==1) ): return 0
            else: return 1
        else:
            if( (ladder>0 and ladder%2==0) or (ladder<0 and ladder%2==1) ): return 1
            else: return 0
    else:
        raise Exception(f'Layer {layer} not recognized.')
        
        
def roc_to_coords(roc, layer, ladder, module):
    '''
    Get the coordinates of a given ROC number inside a module.
    Note: the returned coordinates are relative to the module
          and in the plotting convention with the origin in the lower left corner,
          i.e. the first (y-) coordinate is 0 (lower row) or 1 (upper row),
          and the second (x-) coordinate can go from 0 (left) to 7 (right)
    '''
    
    # determine which numbering schema to use
    schema = get_schema(layer, ladder, module)
    
    # define coordinates
    if schema==0: coords = (0 if roc >= 8 else 1, int(abs(roc - 7.5)))
    else: coords = (1 if roc >= 8 else 0, int(abs(roc - 7.5)))
        
    # return result
    return coords


def coords_to_roc(coords, layer, ladder, module):
    '''
    Get the ROC number from given coordinates.
    Note: the returned coordinates are relative to the module
          and in the plotting convention with the origin in the lower left corner,
          i.e. the first (y-) coordinate is 0 (lower row) or 1 (upper row),
          and the second (x-) coordinate can go from 0 (left) to 7 (right)
    '''
    
    # determine which numbering schema to use
    schema = get_schema(layer, ladder, module)
    
    # define coordinates
    if schema==0: roc = abs(coords[1]-7) if coords[0]==1 else coords[1]+8
    else: roc = abs(coords[1]-7) if coords[0]==0 else coords[1]+8
    
    # return the result
    return roc