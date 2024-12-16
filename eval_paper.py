import torch
import numpy as np
from tqdm.auto import tqdm
#import matplotlib.pyplot as plt

def area_computation(language, video, audio):


    #print(f"norm language= {torch.sum(language ** 2, dim=1)}")
    
    language_expanded = language.unsqueeze(1)  # Shape: (n, 1, dim)

    # Compute the differences for all pairs (i-th language embedding with all j-th video/audio embeddings)
    u = language_expanded - video.unsqueeze(0)  # Shape: (n, n, dim)
    v = language_expanded - audio.unsqueeze(0)  # Shape: (n, n, dim)

    # Compute the norms for u and v
    u_norm = torch.sum(u ** 2, dim=2)  # Shape: (n, n)
    v_norm = torch.sum(v ** 2, dim=2)  # Shape: (n, n)

    # Compute the dot products for all pairs
    uv_dot = torch.sum(u * v, dim=2)  # Shape: (n, n)

    # Calculate the area for all pairs. I remove sqrt calculation
    area = torch.sqrt((u_norm * v_norm) - (uv_dot ** 2)) / 2  # ((u_norm * v_norm) - (uv_dot ** 2))/2
    
    return area       

def volume_computation(language, video, audio):

    batch_size = language.shape[0]

    # Compute all pairwise dot products for language_i @ language_i
    # Shape: (batch_size, batch_size) for all language pairs
    li = torch.einsum('bi,bi->b', language, language).unsqueeze(0).expand(batch_size, batch_size)


    # Compute pairwise dot products for language_i @ video_j and language_i @ audio_j (shape: [batch_size, batch_size])
    vi = language@video.T
    ai = language@audio.T
    # Compute all pairs for video_j @ video_j, video_j @ audio_j, and audio_j @ audio_j
    # Shape: (batch_size)
    vv = torch.einsum('bi,bi->b', video, video)
    va = torch.einsum('bi,bi->b', video, audio)
    aa = torch.einsum('bi,bi->b', audio, audio)

    # Reshape vv, va, and aa to match the shape of vi and ai (i.e., expand along the batch dimension)
    vv = vv.unsqueeze(0).expand(batch_size, -1)
    va = va.unsqueeze(0).expand(batch_size, -1)
    aa = aa.unsqueeze(0).expand(batch_size, -1)

    # Stack the results to form the Gram matrix for each pair
    # Shape: (batch_size, batch_size, 3, 3)
    G = torch.stack([
        torch.stack([li, vi, ai], dim=-1),  # First row of the Gram matrix
        torch.stack([vi, vv, va], dim=-1),  # Second row of the Gram matrix
        torch.stack([ai, va, aa], dim=-1)   # Third row of the Gram matrix
    ], dim=-2)

    # Compute the determinant for each Gram matrix (batch_size, batch_size)
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))

    return res

def main():

    # LOAD VIDEO,LANGUAGE AND AUDIO EMBEDDINGS.
    # EACH OF THEM SHOULD REFER TO THE SAME DATASET AND THEY SHOULD BE EXTRACTED FROM
    # A PRETRAINED EMBEDDING MODEL
    # IS IT IMPORTANT THAT EMBEDDING VIDEO i-TH REFERS TO VIDEO i-TH
    # THE SAME SHOULD HOLD FOR LANGUAGE AND AUDIO
    
    #video = np.load('PATH/TO/EMBEDDINGS/VIDEO.npy')   
    #language = np.load('PATH/TO/EMBEDDINGS/LANGUAGE.npy')  
    #audio = np.load('PATH/TO/EMBEDDINGS/AUDIO.npy')
    
    #NORMALIZE SO THAT EACH OF THEM HAS NORM=1
    #language = normalize( language, axis = 1 )
    #video = normalize(video, axis = 1)
    #audio = normalize(audio, axis = 1)
    language = torch.load("./experiments/449text_features_msrvtt.pt",map_location=torch.device('cpu'))
    video =    torch.load("./experiments/449video_features_msrvtt.pt",map_location=torch.device('cpu'))
    audio =    torch.load("./experiments/449audio_features_msrvtt.pt",map_location=torch.device('cpu'))
    #COMPUTE VIDEO-TEXT SIMILARITY MATRIX
    #sim_matrix = torch.tensor(video @ language.T)
    sim_matrix = torch.tensor(language @ video.T)
    sim_video_text = sim_matrix
    #COMPUTE VIDEO-TO-TEXT AND TEXT-TO-VIDEO SCORES
    print('VT', compute_metrics(sim_matrix))
    print('TV', compute_metrics(sim_matrix.T))

    #COMPUTE AUDIO-TEXT SIMILARITY MATRIX
    sim_matrix = torch.tensor(  language @ audio.T)
    sim_audio_text = sim_matrix
    #COMPUTE AUDIO-TO-TEXT AND TEXT-TO-AUDIO SCORES
    print('AT', compute_metrics(sim_matrix))
    print('TA', compute_metrics(sim_matrix.T))


    #AS A TOY EXAMPLE COMPUTE AREA FOR THREE SIMPLE VECTORS
    u_vectors = language[0] - video[0]
    v_vectors = language[0] - audio[0]
    
    #COMPUTE NORM OF U VECTOR AND V VECTOR
    u_vector_norm =  torch.tensor(u_vectors @ u_vectors.T)
    v_vector_norm =  torch.tensor(v_vectors @ v_vectors.T)

    #COMPUTE DOT PRODUCT BETWEEN U AND V
    uv_vector_dot =  torch.tensor(u_vectors @ v_vectors.T)

    #COMPUTE AREA USING EQ. 2
    area = (u_vector_norm * v_vector_norm) - (uv_vector_dot * uv_vector_dot) / 2 #torch.sqrt( 
    
    print(f"AREA FIRST EXAMPLE: {area}")




    #COMPUTE AREA AMONG ALL POSSIBLE COMBINATIONS OF TEXT VIDEO AND AUDIO EMBEDDINGS
    #SAVE AREA IN A MATRIX FOR METRICS COMPUTATION
    #res = []
#
    #for i in tqdm(range(language.shape[0])):
    #    #FOR EACH TEXTUAL EMBEDDING, COMPARE ONE FIXED TEXTUAL EMBEDDING WITH EACH VISUAL-AUDIO EMBEDDINGS 
    #    l=[]
    #    for j in range(language.shape[0]):
    #        #COMPUTE AREA AMONG i-TH TEXTUAL EMBEDDING AND EACH j-TH VISUAL AND AUDIO EMBEDDINGS
    #        u = language[i] - video[j]
    #        u_norm = torch.tensor(u@ u.T)
#
    #        v = language[i] - audio[j]
    #        v_norm = torch.tensor(v@ v.T)
#
    #        uv_dot = u@v.T
    #        area = torch.sqrt( (u_norm * v_norm) - (uv_dot * uv_dot)) / 2
    #        l.append(area.item())
    #    
    #    
    #    res.append(l)
        
    #AT THE END WE HAVE A MATRIX (#SAMPLES,#SSAMPLES) IN WHICH WE HAVE IN POSITION (i,j)
    #THE AREA FORMED BY THE i-TH TEXTUAL EMBEDDING AND j-TH VISUAL AND AUDIO EMEBDDINGS
    #res = np.array(res)
    #res = torch.from_numpy(res)
    res=volume_computation(language, video, audio)

    #cOMPUTE FINAL METRICS USING SUCH MATRIX
    print("-----")
    print('Triangle Area Results:')
    print("-----")

    #EQ.2
    print('AV AREA', compute_metrics(res,type="area"))
    print('AV AREA transp', compute_metrics(res.T,type="area"))

    #REGULARIZATION USING VIDEO-TEXT SIMILARITY EQ 3
    print('AV AREA + video', compute_metrics(res-sim_video_text,type="area"))
    print('AV AREA + video Transp', compute_metrics((res-sim_video_text).T,type="area"))

    #REGULARIZATION USING AUDIO-TEXT SIMILARITY EQ 3
    print('AV AREA + audio', compute_metrics(res-sim_audio_text,type="area"))
    print('AV AREA + audio Transp', compute_metrics((res-sim_audio_text).T,type="area"))

    #REGULARIZATION USING AUDIO-TEXT SIMILARITY AND VIDEO-TEXT SIMILARITY

    print('AV AREA = video and audio', compute_metrics(res-sim_video_text-0.05*sim_audio_text,type="area"))
    print('AV AREA = video and audio', compute_metrics((res-sim_video_text-0.05*sim_audio_text).T,type="area"))




    eye = torch.eye(res.shape[0])
    sim_vid_diag = sim_video_text*eye

    print('AV AREA + video', compute_metrics(res-sim_vid_diag,type="area"))
    print('AV AREA + video Transp', compute_metrics((res-sim_vid_diag).T,type="area"))

    print(sim_vid_diag[0][0])
    #ABLATION STUDIES ON ALPHA HYPERPARAM
    #TRIES ALL VALUES FROM 0 TO 1 WITH A STEP OF 0.05
    hyperparameter_values = np.arange(0, 1.55, 0.05)
    
    scores = [[],[],[]]
    scores_transp =  [[],[],[]]
    for value in hyperparameter_values:
        tmp = compute_metrics(res-  value * sim_video_text,type="area")
        scores[0].append(tmp['R1'])
        scores[1].append(tmp['R5'])
        scores[2].append(tmp['R10'])

        tmp  = compute_metrics((res- value * sim_video_text).T,type="area")
        scores_transp[0].append(tmp['R1'])
        scores_transp[1].append(tmp['R5'])
        scores_transp[2].append(tmp['R10'])
    
    print(scores)
    print(scores_transp)

    #PLOT RESULTS OF ABLATION STUDIES
    plot_multiple_and_save(hyperparameter_values,scores,["R@1","R@5","R@10"],"./SCORES.png")
    plot_multiple_and_save(hyperparameter_values,scores_transp,["R@1","R@5","R@10"],"./TRANSPOSED_SCORES.png")

    



def plot_multiple_and_save(x, y_lists, labels, filename):
    """
    Plots multiple y lists against a single x list and saves the plot as an image.

    Parameters:
    - x: List of x values.
    - y_lists: List of lists of y values (one list for each line to be plotted).
    - labels: List of labels for each y list.
    - filename: Name of the file to save the plot (e.g., 'plot.png').
    """
    import seaborn as sns
    sns.set_style("dark")

    plt.figure(figsize=(10, 6))
    
    for y, label in zip(y_lists, labels):
        plt.plot(x, y, marker='o', label=label)
    
    plt.xlabel('Alpha values')
    plt.ylabel('Metrics values')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    
    
def compute_metrics(x,type="standard"):
    if type=="standard":
        sx = np.sort(-x, axis=1)
        d = np.diag(-x)
    elif type=="area":
        sx = np.sort(x, axis=1)
        d = np.diag(x)

    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    return metrics

def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

if __name__ == '__main__':
    main()