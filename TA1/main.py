import pygame,math
import neat

import pickle

from helper import getSensorX, getSensorY

pygame.init()
screen = pygame.display.set_mode((1067,600))

pygame.display.set_caption("Car racing")
background_image = pygame.image.load("track.png").convert()
player_image = pygame.image.load("car.png").convert_alpha()

player=pygame.Rect(400,300,20,20)

WHITE=(255,255,255)
xvel=2
yvel=3
angle=0
change=0

distance=2
forward=False

font = pygame.font.Font('freesansbold.ttf', 12)

def newxy(x,y,distance,angle):
  angle=math.radians(angle+90)

  xnew=x+(distance*math.cos(angle))
  ynew=y-(distance*math.sin(angle))

  return xnew,ynew

def checkOutOfBounds(car):
  x = car.x
  y = car.y
  width = car.width
  height = car.height

  if(checkPixel(x,y) or checkPixel(x+width, y) or checkPixel(x, y+height) or checkPixel(x+width, y+height)):
      return True
  
def checkPixel(x, y):
    global screen
    try:
        color = screen.get_at((x, y))
    except:
        return 1
    if(color == (137,137,137,255)):
        return 0
    return 1

def getSensorsData(car, angle):
    global screen
    # Create margin variable i.e distance of sensor from the car
    margin = 55
    
    # Get Center of the car in x variable 
    x = car.x + car.w/2
    # get car.y in y variable
    y = car.y 
    
    # Create a variable sensorAngle which tells at which direction sensor will be placed from center of the car
    sensorAngle = -10
    
    # Use getSensorX and getSensorY helper functions to get newX and newY i.e location of the sensor
    newX = getSensorX(x, angle, sensorAngle, margin)
    newY = getSensorY(y, angle, sensorAngle, margin)      
                
    # Draw a rectangle at given x and y location of the sensor            
    pygame.draw.rect(screen,(0, 255,0), [newX, newY, 5, 5])
        
    
gen=0
angle =0

def eval_fitness(generation, config):
    global angle, gen, forward, change
    gen = gen+1
    genomeCount = 1

    print("Generation:", gen, "Total", len(generation) )
    
    for gid, genome in generation:
        
        genome.fitness = 0 
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        infoText = font.render('Generation'+ str(gen)+  ' genomecount: (' +str(len(generation))+") :" +str(genomeCount), True, (255,255,0))

        while True:
          screen.blit(background_image,[0,0])
          screen.blit(infoText, (420, 20))
          
          for event in pygame.event.get():
            if event.type == pygame.QUIT:
              pygame.quit()
              
            if event.type == pygame.KEYDOWN:
               if event.key == pygame.K_LEFT:
                  change = 5
               if event.key ==pygame.K_RIGHT:
                change = -5 

               if event.key == pygame.K_UP:
                forward = True
                
            if event.type == pygame.KEYUP:
              if event.key ==pygame.K_LEFT or event.key == pygame.K_RIGHT:
                  change = 0
              if event.key == pygame.K_UP:
                forward = False 
            
         
          if forward:
              player.x,player.y=newxy(player.x, player.y, 3, angle)  
                          
          if(checkOutOfBounds(player)):
              player.x = 170
              player.y = 300
              angle =0
              genomeCount = genomeCount +1
              break
          
          angle = angle + change
          
          newimage=pygame.transform.rotate(player_image,angle) 
          pygame.draw.rect(screen,(0, 255, 0), player)
          screen.blit(newimage ,player)
            
          forward = True
          change = 0
          
          # SA1:- Call getSensorData(player, angle)
          getSensorsData(player, angle)          
          
          output = net.activate((player.x, player.y))
          
          if output[0] > 0.65:
             change = 3
          if output[1] > 0.65:
             change = -3
                    
          pygame.display.update()
          pygame.time.Clock().tick(30)
  
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,'config-feedforward.txt')  
p = neat.Population(config)
winner = p.run(eval_fitness,10) 
