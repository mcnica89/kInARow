import numpy as np
import random
import pygame
import math
from pygame.locals import *

SQUARE_SIZE = 100



from timeit import default_timer as timer

FULL_BOARD = np.iinfo(np.uint64).max #the maximum integer corresponds to a full bitboard
EMPTY_BOARD = np.uint64(0)

N_COL = 7
N_ROW = 6
assert N_COL*N_ROW <= 64 #since we are using bitboards with 64 bits


screen = pygame.display.set_mode((SQUARE_SIZE*N_COL,SQUARE_SIZE*(N_ROW+1)))
pygame.init()

RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
DARKBLUE = (0,0,128)
WHITE = (255,255,255)
BLACK = (0,0,0)
PINK = (255,200,200)
KEYS = [K_0,K_1,K_2,K_3,K_4,K_5,K_6,K_7,K_8,K_9]

#Conventions:
# ALLCAPS = const
# _varname = temp
# "s" = "state"
# "a" = "action"
# CamelCase = functions
#Other stuff:
# The bitboards are setup so that 1 is printed in the top-left corner when you print it
# The bottom right corner has the largest value on the board.

def PrintGame(s1,s2):
        #print the board in text given the bitboards for player 1 and player 2
        pieces_s1 = BitboardToArray(s1)
        pieces_s2 = BitboardToArray(s2)

        print("=="*N_COL)
        for i in range(N_ROW-1,-1,-1):
            for j in range(N_COL-1,-1,-1):
                if pieces_s1[i,j] == True:
                    print("X ",end='')
                elif pieces_s2[i,j] == True:
                    print("O ",end='')
                else:
                    print("- ",end='')
            print("")
        print("=="*N_COL)
        print(' '.join(str(i) for i in range(N_COL)))
        print("=="*N_COL)
        print("")

def PrintState(s,player=0):
    if player == 0:
        PrintGame(s,0)
    else:
        PrintGame(0,s)

def PrintState(s,player=0):
    if player == 0:
        PrintGame(s,0)
    else:
        PrintGame(0,s)

def PrintStateX(s):
    PrintGame(s,0)


def PrintStateO(s):
    PrintGame(0,s)


def BitboardToArray(s_bitboard):
    _iter = (char == '1' for char in np.binary_repr(s_bitboard,N_ROW*N_COL))
    return np.fromiter(_iter, dtype=np.bool).reshape((N_ROW,N_COL))

def PygameDraw(s0,s1):
        #Draws the board in pygame
        screen.fill(WHITE)
        for i in range(N_COL):
            font = pygame.font.Font(None, 36)
            text = font.render(str(i), 1, BLACK)
            textrect = text.get_rect(center=((i+0.5)*SQUARE_SIZE,SQUARE_SIZE/2))
            screen.blit(text,textrect)
        for i in range(N_COL):
            for j in range(N_ROW):
                pygame.draw.rect(screen, BLACK, (i*SQUARE_SIZE,(N_ROW-j)*SQUARE_SIZE,SQUARE_SIZE,SQUARE_SIZE), 2)

        p0 = BitboardToArray(s0)
        p1 = BitboardToArray(s1)

        for i in range(N_COL):
            for j in range(N_ROW):
                if p0[j,N_COL - 1 - i] == True:
                    pygame.draw.circle(screen, RED, (i*SQUARE_SIZE+SQUARE_SIZE//2,(N_ROW-j)*SQUARE_SIZE+SQUARE_SIZE//2),(SQUARE_SIZE//2), 0)
                if p1[j,N_COL - 1 - i] == True:
                    pygame.draw.circle(screen, BLACK, (i*SQUARE_SIZE+SQUARE_SIZE//2,(N_ROW-j)*SQUARE_SIZE+SQUARE_SIZE//2),(SQUARE_SIZE//2), 0)
        pygame.display.update()

VAL_MATRIX = np.outer(np.power(np.uint64(2**N_COL),np.arange(N_ROW-1,-1,-1,dtype=np.uint64)),np.power(2,np.arange(N_COL-1,-1,-1,dtype=np.uint64)))
#array that allows easy conversion between bitboards and arrays
def ArrayToBitboard(s_array):
    #inputs a N_COL by N_ROW array and outputs a bitboard (a uint64 whose binary rep is the pieces)
    return np.uint64(np.sum(s_array*VAL_MATRIX))

def IsSubset(s_sub,s_super):
    #Returns TRUE if the bitboard s_sub is a subset of s_super
    #Works if you put in an np array for either s_sub or s_super via broadcasting
    return np.bitwise_or(np.bitwise_not(s_sub),s_super) == FULL_BOARD

def IsNotCovered(s_sub,s_super):
    #Returns TRUE if the bitboard s_sub is entirely on things that s_super doesnt cover.
    #Works if you put in an np array for either s_sub or s_super via broadcasting
    return np.bitwise_and(s_sub,s_super) == EMPTY_BOARD

def CoordToBb(i,j):
    #Returns the bitboard for the location i,j
    return np.uint64(2**(i*N_COL+j))
    #return np.left_shift(np.uint64(1),np.uint(i)*N_COL+np.uint(j))
    #This gives me some kind of bug about the type of i,j

COLUMN_ZERO_PIECES = np.array( [CoordToBb(y,0) for y in range(N_ROW)], dtype=np.uint64)
ALL_PIECES = np.outer(COLUMN_ZERO_PIECES,np.uint64(2**np.arange(N_COL)))
def AllowedMoves(s):
    #Input: s = union of all pieces on board
    #Ouput: A vector of length N_COL with the bitboard for the allowed move from that column (and 0 if no move is allowed there)
    #Works by taking the array ALL_PIECESthen finding the maximum piece in each column that is not covered
    return(np.amax(np.multiply(ALL_PIECES, IsNotCovered(ALL_PIECES,s)),axis=0))

def GenerateKinARow(k):
    return np.array(
        [ np.sum([CoordToBb(x,y+i) for i in range(k)]) for x,y in np.ndindex(N_ROW,N_COL-k+1)] +
        [ np.sum([CoordToBb(x+i,y) for i in range(k)]) for x,y in np.ndindex(N_ROW-k+1,N_COL)] +
        [ np.sum([CoordToBb(x+i,y+i) for i in range(k)]) for x,y in np.ndindex(N_ROW-k+1,N_COL-k+1)] +
        [ np.sum([CoordToBb(x+i,N_COL-1-(y+i)) for i in range(k)]) for x,y in np.ndindex(N_ROW-k+1,N_COL-k+1)])

FOUR_IN_A_LINE = GenerateKinARow(4)
THREE_IN_A_LINE = GenerateKinARow(3)
TWO_IN_A_LINE = GenerateKinARow(2)
SINGLES = np.array( [CoordToBb(y,x) for y in range(N_ROW) for x in range(N_COL)] )
#All possible four in a row states, horiz, vertical, northeast then southeast
def HasConnectFour(s):
    return(np.any(IsSubset(FOUR_IN_A_LINE,np.uint64(s))))

def IsPatternSatisfied(A,B,s0,s1):
    return(IsSubset(A,s0) and IsNotCovered(B,s1))
