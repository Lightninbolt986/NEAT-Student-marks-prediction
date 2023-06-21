
import os
import neat
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
df = pd.read_csv('./student-mat.csv',delimiter=',')
y = [tuple(x[0]) for x in list(zip(df.iloc[: , :-1].values.tolist()))]
x = df.iloc[: , -1].values
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2)
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        for i in range(len(ytrain)):
            if(not xtrain[i]==0):
                output = net.activate(ytrain[i])
                genome.fitness += 100*abs(output[0] - xtrain[i])/xtrain[i]
        genome.fitness /= len(ytrain)
        genome.fitness = -genome.fitness
def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    xpredicted = []
    for i in range(len(ytest)):
        output = winner_net.activate(ytest[i])[0]
        xpredicted.append(output)
        print("input {!r}, expected output {!r}, got {!r}".format(ytest[i], xtest[i] , output))
    xpredicted = np.array(xpredicted)
    p = pd.DataFrame(list(zip(*list(zip(*ytest)),xtest,xpredicted, abs(xpredicted-xtest),100*abs(xpredicted-xtest)/xtest)), columns = [x for x in df]+['predicted','absolute error', '%absolute error'])
    
    print(p.describe().to_string())
    print(p.to_string())
    node_names = {-1:'school',-2:'sex',-3:'age',-4:'address',-5:'famsize',-6:'Pstatus',-7:'Medu',-8:'Fedu',-9:'Mjob',-10:'Fjob',-11:'reason',-12:'guardian',-13:'traveltime',-14:'studytime',-15:'failures',-16:'schoolsup',-17:'famsup',-18:'paid',-19:'activities',-20:'nursery',-21:'higher',-22:'internet',-23:'romantic',-24:'famrel',-25:'freetime',-26:'goout',-27:'Dalc',-28:'Walc',-29:'health',-30:'absences',-31:'G1',-32:'G2',0:'G3'}

    """    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=False)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
   """  """
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)"""
if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    
    run(config_path)

