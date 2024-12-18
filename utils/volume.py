import torch
import numpy as np

# * * * *  * * * *  * * * *   *       *   
# *        *     *  *     *   * *   * *   
# *   * *  * * *    * * * *   *   *   *   
# *     *  *     *  *     *   *       *   
# * * * *  *     *  *     *   *       *   

# THIS IS THE CORE PY CODE OF GRAM FRAMEWORK



def volume_computation3(language, video, audio):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 4x4
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    - subtitles (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing subtitle features.
    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    # Compute pairwise dot products for language with itself (shape: [batch_size1, 1])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with video, audio, and subtitles (shape: [batch_size1, batch_size2])
    lv = language@video.T
    la = language@audio.T

    # Compute pairwise dot products for video, audio, and subtitles with themselves and with each other
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    va = torch.einsum('bi,bi->b', video, audio).unsqueeze(0).expand(batch_size1, -1)
    aa = torch.einsum('bi,bi->b', audio, audio).unsqueeze(0).expand(batch_size1, -1)
    


    # Stack the results to form the Gram matrix for each pair (shape: [batch_size1, batch_size2, 4, 4])
    G = torch.stack([
        torch.stack([ll, lv, la], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, vv, va], dim=-1),  # Second row of the Gram matrix
        torch.stack([la, va, aa], dim=-1)  # Third row of the Gram matrix
    ], dim=-2)

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    #print(res.shape)
    return res


def volume_computation4(language, video, audio, subtitles):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 4x4
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    - subtitles (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing subtitle features.
    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    # Compute pairwise dot products for language with itself (shape: [batch_size1, 1])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with video, audio, and subtitles (shape: [batch_size1, batch_size2])
    lv = language@video.T
    la = language@audio.T
    ls = language@subtitles.T

    # Compute pairwise dot products for video, audio, and subtitles with themselves and with each other
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    va = torch.einsum('bi,bi->b', video, audio).unsqueeze(0).expand(batch_size1, -1)
    aa = torch.einsum('bi,bi->b', audio, audio).unsqueeze(0).expand(batch_size1, -1)
    
    ss = torch.einsum('bi,bi->b', subtitles, subtitles).unsqueeze(0).expand(batch_size1, -1)
    vs = torch.einsum('bi,bi->b', video, subtitles).unsqueeze(0).expand(batch_size1, -1)
    sa = torch.einsum('bi,bi->b', audio, subtitles).unsqueeze(0).expand(batch_size1, -1)

    # Stack the results to form the Gram matrix for each pair (shape: [batch_size1, batch_size2, 4, 4])
    G = torch.stack([
        torch.stack([ll, lv, la, ls], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, vv, va, vs], dim=-1),  # Second row of the Gram matrix
        torch.stack([la, va, aa, sa], dim=-1),  # Third row of the Gram matrix
        torch.stack([ls, vs, sa, ss], dim=-1)   # Fourth row of the Gram matrix
    ], dim=-2)

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    #print(res.shape)
    return res


def volume_computation5(language, video, audio, subtitles, depth):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 4x4
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    - subtitles (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing subtitle features.
    - depth (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing depth features.    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    # Compute pairwise dot products for language with itself (shape: [batch_size1, 1])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with video, audio, and subtitles (shape: [batch_size1, batch_size2])
    lv = language@video.T
    la = language@audio.T
    ls = language@subtitles.T
    ld = language@depth.T

    # Compute pairwise dot products for video, audio, and subtitles with themselves and with each other
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    va = torch.einsum('bi,bi->b', video, audio).unsqueeze(0).expand(batch_size1, -1)
    aa = torch.einsum('bi,bi->b', audio, audio).unsqueeze(0).expand(batch_size1, -1)
    
    
    ss = torch.einsum('bi,bi->b', subtitles, subtitles).unsqueeze(0).expand(batch_size1, -1)
    vs = torch.einsum('bi,bi->b', video, subtitles).unsqueeze(0).expand(batch_size1, -1)
    sa = torch.einsum('bi,bi->b', audio, subtitles).unsqueeze(0).expand(batch_size1, -1)

    dd = torch.einsum('bi,bi->b', depth, depth).unsqueeze(0).expand(batch_size1, -1)
    dv = torch.einsum('bi,bi->b', depth, video).unsqueeze(0).expand(batch_size1, -1)
    da = torch.einsum('bi,bi->b', depth, audio).unsqueeze(0).expand(batch_size1, -1) 
    ds = torch.einsum('bi,bi->b', depth, subtitles).unsqueeze(0).expand(batch_size1, -1)


    # Stack the results to form the Gram matrix for each pair (shape: [batch_size1, batch_size2, 4, 4])
    G = torch.stack([
        torch.stack([ll, lv, la, ls, ld], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, vv, va, vs, dv], dim=-1),  # Second row of the Gram matrix
        torch.stack([la, va, aa, sa, da], dim=-1),  # Third row of the Gram matrix
        torch.stack([ls, vs, sa, ss, ds], dim=-1),   # Fourth row of the Gram matrix
        torch.stack([ld, dv, da, ds, dd], dim=-1)
    ], dim=-2)

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    #print(res.shape)
    return res
