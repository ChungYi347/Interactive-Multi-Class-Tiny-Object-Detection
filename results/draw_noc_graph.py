import os
import os.path as osp
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

def plot_init(fig_size):
    plt.close()
    plt.clf()
    plt.cla()
    plt.tight_layout(pad=0)
    fig = plt.figure(figsize=fig_size)
    return fig

def make_graphs(noc_data_path):
    if noc_data_path in ["Tiny-DOTA-FasterRCNN"]:
        files = ['Ours.json', 'Late_Fusion.json', 'Early_Fusion.json', 'Passthrough.json', 'Faster_R-CNN.json']
    elif noc_data_path in ["Tiny-DOTA-RetinaNet"]:
        files = ['Ours.json', 'Late_Fusion.json', 'Early_Fusion.json', 'Passthrough.json', 'RetinaNet.json']
    else:
        files = sorted(os.listdir(noc_data_path))

    with plt.style.context('seaborn-bright'):
        plt.rcParams['axes.prop_cycle'] = cycler(color=['#E8000B', '#003FFF', '#03ED3A', '#8A2BE2', '#FFC400', '#00D7FF'])
        fig = plot_init((7, 2.7))

        for f_path in files:
            if 'json' not in f_path:
                continue
            data = json.load(open(f'{noc_data_path}/{f_path}'))

            if 'mmap' not in list(data.keys()):
                maps = []
                for p in data:
                    maps.append(data[p]['map'])
                maps = np.array(maps)
                f_name = f_name.replace('_', ' ')
                plt.errorbar(points, maps[1:max_points, -1], label=f_name, linewidth=2)
            else:
                f_name = f_path.split('.')[0]
                max_points = 21
                points = [i+1 for i in range(max_points-1)]
                
                maps = data['mmap']
                yerr = data['std']
                linestyle = 'solid'

                f_name = f_name.replace('_', ' ')

                if len(maps) == max_points-1:
                    plt.errorbar(points, maps[:max_points], yerr=yerr[:max_points], label=f_name, capsize=3)
                else:
                    plt.errorbar(points, maps[1:max_points], yerr=yerr[1:max_points], label=f_name, capsize=3)
            plt.ylabel('mAP')
            # plt.xlabel('Number of Clicks')
            plt.xticks(points
                # rotation=45
            )
            plt.xlim(1,20)

        if noc_data_path in ['Tiny-DOTA-FasterRCNN']:
            plt.legend(loc='upper left', bbox_to_anchor=(-0.08, 1.2), ncol=5, prop={'size': 8.5})
        elif noc_data_path in ['Tiny-DOTA-RetinaNet']:
            plt.legend(loc='upper left', bbox_to_anchor=(-0.06, 1.15), ncol=5, prop={'size': 8.5})
        else:
            plt.legend(ncol=3, prop={'size': 10})
        plt.grid(True)

        plt.savefig(osp.join(noc_data_path, f'total_noc_graph.pdf'), bbox_inches='tight')
        plt.savefig(osp.join(noc_data_path, f'total_noc_graph.png'), bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make total noc graph')
    make_graphs('Tiny-DOTA-FasterRCNN')
    make_graphs('Tiny-DOTA-RetinaNet')
   

