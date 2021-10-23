# Blair Johnson 2021

from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np


def create_embeddings(images):
    '''
    Take an iterable of image candidates and return an iterable of image embeddings.
    '''
    if type(images) != list:
        images = [images]

    extractor = MTCNN()
    encoder = InceptionResnetV1(pretrained='vggface2').eval()
    
    embeddings = []
    for image in images:
        cropped_img = extractor(image)
        embeddings.append(encoder(cropped_img.unsqueeze(0)))

    return embeddings

def candidate_search(candidates, target):
    '''
    Take an iterable of candidates and a target image and determine the best candidate fit
    '''

    cand_embs = create_embeddings(candidates)
    target_embs = create_embeddings(target)[0]

    best_loss = np.inf
    best_candidate = np.inf
    for i,embedding in enumerate(cand_embs):
        loss = np.linalg.norm(target_embs.detach().numpy()-embedding.detach().numpy(), ord='fro')
        
        if loss < best_loss:
            best_loss = loss
            best_candidate = i

    return candidates[i], best_candidate

if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt

    test1 = np.array(Image.open('/home/bjohnson/Pictures/fake_face.jpg'))
    test2 = np.array(Image.open('/home/bjohnson/Pictures/old_face.jpg'))
    test3 = np.array(Image.open('/home/bjohnson/Pictures/young_face.jpg'))
    target = np.array(Image.open('/home/bjohnson/Pictures/profile_pic_lake_louise.png'))
    
    candidates = [test1,test2,test3]
    
    chosen, index = candidate_search(candidates, target)
    print(index)
    #plt.imshow(candidate_search(candidates, target))
