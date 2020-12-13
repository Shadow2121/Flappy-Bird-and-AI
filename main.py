import pygame
import neat
import time 
import os
import random
import cls

pygame.font.init()

## Defining constants
WIN_WIDTH = 575
WIN_HEIGHT = 800
GEN = 0
BIRDS_ALIVE = 0

## Importing Imgs
BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))
STAT_FONT = pygame.font.SysFont("comicsans", 50)


## Drowing Main game window
def draw_win(win, birds, pipes, base, score, b_alive):
    global GEN
    win.blit(BG_IMG, (0,0))

    for p in pipes:
        p.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Generation: " + str(GEN), 1, (255, 255, 255))
    win.blit(text, (10, 10))
    text = STAT_FONT.render("Alive: " + str(b_alive), 1, (255, 255, 255))
    win.blit(text, (10, 50))

    base.draw(win)
    for bird in birds:
        bird.draw(win)
    pygame.display.update()


def main(genomes, config):
    global GEN
    GEN += 1
    birds = []
    ge = []
    nets = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(cls.Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    base = cls.Base(730)
    pipes = [cls.Pipe(800)]

    score = 0

    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption('Flappy Bird')
    clock = pygame.time.Clock()

    run = True
    ## Main game loop
    while run:
        clock.tick(30)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:                
                run = False
                pygame.quit()
                quit()
            # if e.type == pygame.KEYDOWN:
            #     if e.key == pygame.K_SPACE:
            #         bird.jump()

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].hight), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.jump()

        # bird.move()
        rem = []
        add_pipe = False
        for pipe in pipes:
            for bird in birds:
                if pipe.colloide(bird):
                    ge[birds.index(bird)].fitness -= 1
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()
            
        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(cls.Pipe(630))

        for r in rem:
            pipes.remove(r)

        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= 730 or bird.y < -50:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        base.move()
        draw_win(win, birds, pipes, base, score, len(birds))


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    state = neat.StatisticsReporter()
    p.add_reporter(state)

    winner = p.run(main, 50)

    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)