import torchvision.transforms as T
import torch
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform  
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision.transforms as transforms

def get_train_transform(resize_crop_size = 256,
                  mean = [0.4139, 0.4341, 0.3482, 0.5263],
                  std = [0.0010, 0.0010, 0.0013, 0.0013]
                  ):

    augmentation = A.Compose(
        [
            A.RandomResizedCrop(height=resize_crop_size, width=resize_crop_size),
            A.RandomBrightnessContrast(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.GaussianBlur(),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    def transform(sample):
        image = sample["image"].numpy().transpose(1,2,0)
        point = sample["point"]

        image = augmentation(image=image)["image"]
        point = coordinate_jitter(point)

        return dict(image=image, point=point)

    return transform

def get_pretrained_s2_train_transform(resize_crop_size = 256):
    augmentation = T.Compose([
        T.RandomCrop(resize_crop_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.GaussianBlur(3),
    ])

    def transform(sample):
        image = sample["image"] / 255.0
        point = sample["point"]

        #B10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
        #image = np.concatenate([image[:10], B10, image[10:]], axis=0)
        image = torch.tensor(image)

        image = augmentation(image)

        point = coordinate_jitter(point)

        return dict(image=image, point=point)

    return transform

def get_s2_train_transform(resize_crop_size = 256):
    augmentation_image = T.Compose([
        T.RandomCrop(resize_crop_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.GaussianBlur(3),
    ])

    def augmentation_imageVec(image):
        image = np.array(image)
        random_col_index = np.random.randint(0, image.shape[1])
        random_column = image[:, random_col_index]
        return torch.tensor(random_column, dtype=torch.float32)

    def transform(sample):

        # Process images
        image = sample["image"]

        if len(image.shape) == 2: #Vectorised image
            image = augmentation_imageVec(image)

        elif image.shape[0] == 12: # S2 image
            image = image / 255
            B10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
            image = np.concatenate([image[:10], B10, image[10:12]], axis=0)
            image = torch.tensor(image, dtype=torch.float32)
            image = augmentation_image(image)

        else: # RGB image
            image = image / 255
            image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)
            image = augmentation_image(image)

        # Process the points
        point = sample["point"]
        point = coordinate_jitter(point)

        # Process the OSM data
        osm = create_3d_tensor_from_osm_data(sample["osm"]).as_subclass(ExtendedTensor).normalized([48, 1048, 377, 189, 17, 10])

        return dict(image=image, point=point, osm=osm)

    return transform

def mv_noaugment(resize_crop_size = 256):

    augmentation = T.Compose([
        T.RandomCrop(resize_crop_size)
    ])

    def transform(sample):

        if sample['gs'] is not None:
            gs = sample['gs']
            gs = gs / 255
            gs = torch.tensor(gs, dtype=torch.float32).permute(2,0,1)
        else:
            gs = torch.empty((0), dtype=torch.float32)
        
        if sample['osm'] is not None:
            osm = sample['osm']
            osm = create_3d_tensor_from_osm_data(osm).as_subclass(ExtendedTensor).normalized([48, 1048, 377, 189, 17, 10])
        else:
            osm = torch.empty((0), dtype=torch.float32)

        if sample['s2'] is not None:
            s2 = sample['s2']
            s2 = s2 / 255
            B10 = np.zeros((1, *s2.shape[1:]), dtype=s2.dtype)
            s2 = np.concatenate([s2[:10], B10, s2[10:12]], axis=0)
            s2 = torch.tensor(s2, dtype=torch.float32)
            s2 = augmentation(s2)
        else:
            s2 = torch.empty((0), dtype=torch.float32)
        
        if sample['fm'] is not None:
            fm = sample['fm']
        else:
            fm = torch.empty((0), dtype=torch.float32)

        point = sample["point"]

        return dict(gs=gs, osm=osm, s2=s2, fm=fm, point=point)

    return transform

def coordinate_jitter(
        point,
        radius=0.005
    ):
    return point + torch.rand(point.shape) * radius

class ExtendedTensor(torch.Tensor):
    def normalized(self, divisors):
        # Convert divisors to a tensor if it's a list
        if isinstance(divisors, list):
            divisors = torch.tensor(divisors, dtype=torch.float32)

        # Ensure divisors are in the same device as the input tensor
        divisors = divisors.to(self.device)

        # Reshape divisors to match the shape of the input tensor for broadcasting
        divisors = divisors.view(-1, 1, 1)

        # Divide each channel by the corresponding divisor
        normalized_tensor = self / divisors

        # Convert back to regular torch.Tensor
        return torch.Tensor(normalized_tensor)

def create_3d_tensor_from_osm_data(osm_data, columns_of_interest=['health', 'shops', 'tourism', 'food_and_drink', 'education', 'public_services']):
    """
    Create a 3D tensor from OSM data based on specified columns of interest.

    Args:
        osm_data (pd.DataFrame): The OSM data containing 'i', 'j' coordinates and columns of interest.
        columns_of_interest (list): List of column names to be included in the 3D tensor.

    Returns:
        torch.Tensor: A 3D tensor where each slice along the first dimension corresponds to one of the specified columns.
    """
    # Drop the 'h3_address' column if it exists
    if 'h3_address' in osm_data.columns:
        osm_data = osm_data.drop(columns=['h3_address'])

    # Extract the relevant columns and coordinates
    osm_data_select = osm_data[['i', 'j'] + columns_of_interest].copy()

    # Normalize the coordinates to start from 0
    osm_data_select.loc[:, 'i'] = osm_data_select['i'] - osm_data_select['i'].min()
    osm_data_select.loc[:, 'j'] = osm_data_select['j'] - osm_data_select['j'].min()

    # Determine the dimensions of the matrix
    max_i = osm_data_select['i'].max()
    max_j = osm_data_select['j'].max()

    # Create a 3D tensor to hold the matrices
    num_channels = len(columns_of_interest)
    tensor_3d = np.zeros((num_channels, max_i + 1, max_j + 1))

    # Populate the 3D tensor
    for k, column in enumerate(columns_of_interest):
        for _, row in osm_data_select.iterrows():
            i, j, value = int(row['i']), int(row['j']), row[column]
            tensor_3d[k, i, j] = value

    # Convert the 3D numpy array to a PyTorch tensor
    tensor_3d = torch.tensor(tensor_3d, dtype=torch.float32)

    return tensor_3d