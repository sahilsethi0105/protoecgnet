import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from functools import reduce
import operator as op
from backbones import (
    resnet18, resnet34, resnet50, 
    resnet101, resnet152
)

def prototype_loss2d(logits, y_true, model, similarity_scores, class_weights, 
                   lam_clst, lam_sep, lam_spars, lam_div, lam_cnrst, use_contrastive=True):
    """
    Prototype-based loss function for ProtoECGNet.
    - Uses the precomputed similarity scores from the model's forward pass.
    """

    # Ensure all tensors are on the same device
    device = model.prototype_vectors.device
    y_true = y_true.to(device).float()
    logits = logits.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    similarity_scores = similarity_scores.to(device)

    # Expand prototype_class_identity to match batch dimension
    prototypes_of_correct_class = model.prototype_class_identity.unsqueeze(0).expand(y_true.shape[0], -1, -1).to(device)

    # Mask correct prototypes using ground-truth labels
    prototypes_of_correct_class = prototypes_of_correct_class * y_true.unsqueeze(1)

    # Compute activations for correct class prototypes
    correct_class_prototype_activations, _ = torch.max(similarity_scores.unsqueeze(-1) * prototypes_of_correct_class, dim=1)
    
    # Compute activations for incorrect class prototypes
    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
    incorrect_class_prototype_activations, _ = torch.max(similarity_scores.unsqueeze(-1) * prototypes_of_wrong_class, dim=1)

    # Clustering Loss (Encourage correct prototypes to activate)
    clst_loss = -torch.mean(correct_class_prototype_activations)

    # Separation Loss (Encourage incorrect prototypes to be inactive)
    sep_loss = torch.mean(incorrect_class_prototype_activations)

    # Sparsity Loss (Encourage activation of only a few prototypes)
    spars_loss = torch.mean(torch.clamp(similarity_scores, min=0).sum(dim=1))

    # Orthogonality Loss (Encourage diverse prototypes)
    div_loss = model.get_prototype_orthogonalities()
    div_loss = div_loss / (model.num_prototypes ** 2) 

    if use_contrastive:
        # Get prototype vectors (P, C, 1, T) → flatten to (P, D)
        prototypes = model.prototype_vectors.view(model.num_prototypes, -1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)

        # Cosine similarity matrix between all prototypes
        sim_matrix = torch.mm(prototypes_norm, prototypes_norm.T)  # (P, P)
        sim_matrix = torch.clamp(sim_matrix, min=-1 + 1e-6, max=1 - 1e-6)

        # Co-occurrence mask
        P_assign = model.prototype_class_identity.float()  # (P, C)
        L_cooccur = model.label_cooccurrence.to(device)
        
        # Positive similarity weight: co-occurrence between class pairs
        pos_mask = torch.matmul(torch.matmul(P_assign, L_cooccur), P_assign.T)  # (P, P)
        neg_mask = 1 - pos_mask  # Encourage separation otherwise

        # Normalize the masks
        pos_norm = pos_mask / (pos_mask.sum() + 1e-6)
        neg_norm = neg_mask / (neg_mask.sum() + 1e-6)

        # Apply contrastive loss
        pos_loss = torch.sum(sim_matrix * pos_norm)  # Encourage similarity for co-occurring classes
        neg_loss = torch.sum(sim_matrix * neg_norm)   # Discourage similarity otherwise

        cnrst_loss = - (pos_loss - neg_loss) / (model.num_prototypes ** 0.5)
    else:
        cnrst_loss = 0

    # Multi-label Classification Loss 
    classification_loss = F.binary_cross_entropy_with_logits(logits, y_true, pos_weight=class_weights)
    
    print(f"Class loss: {classification_loss}")
    print(f"Clst loss: {lam_clst*clst_loss}")
    print(f"Sep loss: {lam_sep*spars_loss}")
    print(f"Spars loss: {lam_spars*spars_loss}")
    print(f"Div loss: {lam_div*div_loss}")
    print(f"Cnrst loss: {lam_cnrst*cnrst_loss}")

    return classification_loss + lam_clst * clst_loss + lam_sep * sep_loss + lam_spars * spars_loss + lam_div * div_loss + lam_cnrst * cnrst_loss

class ProtoECGNet2D(nn.Module):
    def __init__(self, backbone="resnet18", num_classes=5, single_class_prototype_per_class=5, joint_prototypes_per_border=0, proto_dim=512, proto_time_len=3, 
                prototype_activation_function="log", latent_space_type="l2", add_on_layers_type="other", class_specific=False, last_layer_connection_weight=None, m=None, dropout=0, 
                custom_groups=True, label_set='all', pretrained_weights=None):
   
        super().__init__()
        self.joint_prototypes_per_border = joint_prototypes_per_border
        self.single_class_prototype_per_class = single_class_prototype_per_class
        self.num_classes = num_classes

        num_class_pairs = self.ncr(num_classes, 2)  # Compute number of unique class pairs
        self.num_prototypes = (self.single_class_prototype_per_class * self.num_classes) + (self.joint_prototypes_per_border * num_class_pairs)

        self.prototype_shape = (self.num_prototypes, proto_dim, 1, proto_time_len)
        self.proto_time_len = proto_time_len 
        self.topk_k = 5 # number of sliding windows used to determine similarity score

        self.prototype_activation_function = prototype_activation_function
        self.class_specific=class_specific 
        self.epsilon = 1e-4 
        self.custom_groups = custom_groups
        self.label_set = label_set
        self.m = m 
        self.relu_on_cos = False # unused
        self.input_vector_length = int((self.prototype_shape[1] * self.prototype_shape[3]) ** 0.5)
        self.last_layer_connection_weight = last_layer_connection_weight 
        
        self.latent_space_type = latent_space_type

        try:
            if self.custom_groups:
                if self.label_set == "1":
                    path = "/gpfs/data/bbj-lab/users/sethis/experiments/preprocessing/label_cooccur_Cat1.pt"
                elif self.label_set == "3":
                    path = "/gpfs/data/bbj-lab/users/sethis/experiments/preprocessing/label_cooccur_Cat3.pt"
                elif self.label_set == "4":
                    path = "/gpfs/data/bbj-lab/users/sethis/experiments/preprocessing/label_cooccur_Cat4.pt"
                else:
                    path = "/gpfs/data/bbj-lab/users/sethis/experiments/preprocessing/label_cooccur.pt"
            else:
                path = "/gpfs/data/bbj-lab/users/sethis/experiments/preprocessing/label_cooccur_all.pt"

            cooc_matrix = torch.load(path)
            self.label_cooccurrence = cooc_matrix.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            print(f"[INFO] Loaded label co-occurrence matrix from {path} with shape {self.label_cooccurrence.shape}")
        except Exception as e:
            self.label_cooccurrence = None
            print(f"[WARNING] Failed to load label co-occurrence matrix: {e}")


        # Select backbone feature extractor
        print(f"Initializing ProtoECGNet2D with backbone {backbone}...")
        self.feature_extractor, backbone_only = self._get_backbone(backbone, num_classes, dropout, pretrained_weights)

        feature_dim = proto_dim

        # Even if loading state_dict() from full model checkpoint, have to initialize all components
        if hasattr(self.feature_extractor, "flatten"):
            self.feature_extractor.flatten = nn.Identity()  # Remove flattening layer
        if hasattr(self.feature_extractor, "fc"):
            self.feature_extractor.fc = nn.Identity()  # Remove classifier head
        if hasattr(self.feature_extractor, "conv_reduce"):
            self.feature_extractor.conv_reduce = nn.Identity()  # Remove conv_reduce if present

        self.prototype_vectors = nn.Parameter(torch.randn(*self.prototype_shape), requires_grad=True)
        print(f"Initialized prototype_vectors with shape: {self.prototype_shape}")
        self.ones = nn.Parameter(torch.ones_like(self.prototype_vectors), requires_grad=False) #unused
        self.classifier = nn.Linear(self.num_prototypes, num_classes, bias=False)

        # Initialize Prototype Class Assignments
        self.register_buffer(
            "prototype_class_identity",
            self._create_prototype_labels(num_classes, single_class_prototype_per_class, joint_prototypes_per_border)
        )
        self.prototype_class_identity = self.prototype_class_identity.to(self.prototype_vectors.device)

        self._initialize_add_on_layers(add_on_layers_type, feature_dim)
        self._initialize_weights()
        
        if backbone_only:
            print("Reset prototypes and classifier (Backbone-only checkpoint detected).")
        elif (pretrained_weights is not None):
            # Full model checkpoint detected, load everything
            print(f"Now loading full model from {pretrained_weights}")
            checkpoint = torch.load(pretrained_weights, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
            if 'state_dict' in checkpoint:  # Extract if it's a Lightning checkpoint
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint  # Assume it's already just weights
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("model.", "")  # Remove extra prefix
                new_state_dict[new_key] = v
            if 'criterion.pos_weight' in new_state_dict:
                #print("Removing 'criterion.pos_weight' from checkpoint.")
                del new_state_dict["criterion.pos_weight"]

            # Detect number of prototypes in the checkpoint
            if "prototype_vectors" in new_state_dict:
                loaded_prototypes = new_state_dict["prototype_vectors"].shape[0]
                print(f"Checkpoint contains {loaded_prototypes} prototypes. Adjusting model initialization.")
                self.num_prototypes = loaded_prototypes
                self.prototype_shape = (self.num_prototypes, proto_dim, 1, proto_time_len)

            #Adjust shape of prototype_vectors and classifier to match checkpoint
            self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)
            self.prototype_vectors = nn.Parameter(torch.randn(*self.prototype_shape), requires_grad=True)
            self.classifier = nn.Linear(self.num_prototypes, self.num_classes, bias=False)
            if self.num_prototypes == ((self.single_class_prototype_per_class * self.num_classes) + (self.joint_prototypes_per_border * num_class_pairs)):
                self._create_prototype_labels(num_classes, single_class_prototype_per_class, joint_prototypes_per_border)
            else: 
                self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)

            # print(f"[DEBUG] Checkpoint Keys: {list(new_state_dict.keys())}") 
            # if "prototype_class_identity" in new_state_dict:
            #     print(f"[DEBUG] Loaded Prototype Class Identity Matrix Shape: {self.prototype_class_identity.shape}")
            #     print(f"[DEBUG] Loaded Prototype Class Identity Matrix:\n{self.prototype_class_identity.cpu().numpy()}")
            # else:
            #     print("[WARNING] No `prototype_class_identity` found in checkpoint! Using initialized values.")

            self.load_state_dict(new_state_dict, strict=True)

    def _create_prototype_labels(self, num_classes, single_class_prototype_per_class, joint_prototypes_per_border):
        """Assigns class identities to prototypes (single-class & dual-class) and prints debugging info."""
        identity_matrix = torch.zeros((num_classes * single_class_prototype_per_class) + 
                                    (len(list(combinations(range(num_classes), 2))) * joint_prototypes_per_border), num_classes)

        # Debug tracking
        single_class_counts = {j: 0 for j in range(num_classes)}
        dual_class_counts = {pair: 0 for pair in combinations(range(num_classes), 2)}

        # Assign Single-Class Prototypes
        for j in range(num_classes):
            for k in range(single_class_prototype_per_class):
                idx = j * single_class_prototype_per_class + k
                identity_matrix[idx, j] = 1
                single_class_counts[j] += 1  # Track assignment

        # Assign Joint-Class Prototypes
        class_combinations = list(combinations(range(num_classes), 2))
        dual_start_idx = single_class_prototype_per_class * num_classes  # Start index for dual-class prototypes

        for i, (c1, c2) in enumerate(class_combinations):
            for k in range(joint_prototypes_per_border):
                idx = dual_start_idx + i * joint_prototypes_per_border + k
                identity_matrix[idx, c1] = 1
                identity_matrix[idx, c2] = 1
                dual_class_counts[(c1, c2)] += 1  # Track assignment

        # Debugging Prints
        # print(f"[DEBUG] Identity matrix shape: {identity_matrix.shape}")
        # print(f"[DEBUG] Total single-class prototypes assigned: {sum(single_class_counts.values())}")
        # print(f"[DEBUG] Total dual-class prototypes assigned: {sum(dual_class_counts.values())}")
        # print(f"[DEBUG] Single-class prototype distribution per class: {single_class_counts}")
        # print(f"[DEBUG] Dual-class prototype distribution per border: {dual_class_counts}")

        return identity_matrix
        
    def _initialize_add_on_layers(self, add_on_layers_type, feature_dim):
        """Initializes add-on layers before prototype matching."""
        
        if add_on_layers_type == "identity":  # No transformations, direct prototype matching
            self.add_on_layers = nn.Identity()

        elif add_on_layers_type == "linear":  # Feature transformation while keeping (batch, 512, 1, 32)
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=(1, 1)),
                nn.Sigmoid()
            )

        elif add_on_layers_type == "other":  # Refinement with convolutions
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=(1, 1)),
                nn.Sigmoid()
            )

    @staticmethod
    def ncr(n, r):
        """Computes n choose r (combinations)."""
        r = min(r, n - r)
        numer = reduce(op.mul, range(n, n - r, -1), 1)
        denom = reduce(op.mul, range(1, r + 1), 1)
        return numer // denom

    def _initialize_weights(self):
        """Initializes weights for the model"""
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):  
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Set correct and incorrect class connections in classifier
        self.set_last_layer_incorrect_connection()

    def set_last_layer_incorrect_connection(self, incorrect_class_connection=-0.5):
        """Ensures classifier weight connections are set correctly"""
        if not hasattr(self, "prototype_class_identity"):
            raise ValueError("[ERROR] prototype_class_identity not found. Ensure it's initialized before calling this method.")

        correct_class_connection = self.last_layer_connection_weight
        positive_one_weights_locations = self.prototype_class_identity.T  # (num_classes, num_prototypes)
        negative_one_weights_locations = 1 - positive_one_weights_locations  # Inverse mask for incorrect class connections

        self.classifier.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations +
            incorrect_class_connection * negative_one_weights_locations
        )

        print(f"Updated classifier weight matrix: Correct = {correct_class_connection}, Incorrect = {incorrect_class_connection}")


    def _get_backbone(self, backbone, num_classes, dropout, pretrained_weights):
        """Selects and initializes a 2D backbone, loading full model if available."""
        
        backbones = {
            "resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50,
            "resnet101": resnet101, "resnet152": resnet152
        }
        
        if backbone not in backbones:
            raise ValueError(f"Unsupported 2D backbone: {backbone}")

        # Initialize model
        model = backbones[backbone](num_classes=num_classes, dropout=dropout)
        backbone_only = False 
        
        # Load pre-trained weights if provided
        if pretrained_weights is not None: 
            checkpoint = torch.load(pretrained_weights, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)

            if 'state_dict' in checkpoint:  # Extract if it's a Lightning checkpoint
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint  # Assume it's already just weights
            
            # Check if checkpoint contains prototype & classifier layers
            contains_prototypes = any("prototype_vectors" in k for k in state_dict.keys())
            contains_classifier = any("classifier" in k for k in state_dict.keys())

            if contains_prototypes and contains_classifier:
                print(f"Full ProtoECGNet2D checkpoint detected. Skipping backbone-only loading...")
                backbone_only = False
            else:
                # Only load feature extractor weights, keeping prototypes/classifier unchanged
                print(f"Loading backbone-only weights from {pretrained_weights} (Prototypes & Classifier Unchanged)...")
                model.load_state_dict(state_dict, strict=False)
                backbone_only = True 

        else:
            print("No pretrained weights provided")

        return model, backbone_only
    
    def conv_features(self, x):
        """Extract feature representations from input."""
        x = self.feature_extractor(x)
        #print(f"[DEBUG] Feature Extractor Raw Output Shape: {x.shape}")  # Should be (batch, feature_dim)

        if not isinstance(self.add_on_layers, nn.Identity):
            x = self.add_on_layers(x)  
            #print(f"[DEBUG] After Add-On Layers Shape: {x.shape}")

        return x  # Ensure (batch, feature_dim)
    
    def prototype_distances(self, x, prototypes_of_wrong_class=None):
        """
        Computes prototype distances using L2 convolution or cosine activation.
        Applies top-k pooling within activations.

        Returns:
        - min_distances (for L2) OR max activations (for cosine).
        - similarity scores after top-k pooling.
        """
        conv_features = self.conv_features(x)  # Extract feature map

        if self.latent_space_type == "l2":
            distances = self._l2_convolution(conv_features)  # Compute L2 distances
            min_distances = distances  
            prototype_activations = self.distance_2_similarity(min_distances)  # Convert to similarity

        elif self.latent_space_type == "arc":
            activations, marginless_activations = self.cos_activation(conv_features, prototypes_of_wrong_class)
            
            # Top-k pooling applied across time-windows
            top_k_activations, _ = torch.topk(activations, self.topk_k, dim=-1)
            prototype_activations = F.avg_pool1d(top_k_activations, kernel_size=top_k_activations.shape[-1]).squeeze(-1)

            min_distances = None  # Not used in cosine similarity mode

        else:
            raise ValueError(f"Unsupported latent space type: {self.latent_space_type}")

        return min_distances, prototype_activations
        
    def distance_2_similarity(self, distances):
        """
        Converts prototype distances to similarity scores (Only for L2).

        Returns:
        - similarities: (batch, num_prototypes) after transformation.
        """
        if self.prototype_activation_function == "log":
            similarities = torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == "linear":
            similarities = -distances
        else:
            similarities = self.prototype_activation_function(distances)

        # Apply top-k pooling
        if self.topk_k > 1:
            topk_vals, _ = torch.topk(similarities, k=self.topk_k, dim=-1)  # (batch, topk_k)
            similarities = torch.mean(topk_vals, dim=-1)  # Aggregate top-k values
        else:
            similarities, _ = torch.max(similarities, dim=-1)

        return similarities  # (batch, num_prototypes)
        
    
    def forward(self, x, prototypes_of_wrong_class=None):
        """Forward pass with feature extraction, prototype similarity, and classification."""

        features = self.conv_features(x)  # Extract feature representations
        #] Feature Extractor Output Shape: {features.shape}")  # Expected: (batch, feature_dim)

        if self.latent_space_type == "l2":
            distances = self._l2_convolution(features)  # Compute L2 distances
            #print(f"[DEBUG] L2 Distance Shape: {distances.shape}")  # Expected: (batch, num_prototypes)
            
            min_distances = distances #Since doing global prototypes, just use distances

            #print(f"[DEBUG] Min Distances Shape: {min_distances.shape}")  # Expected: (batch, num_classes)

            # Convert distances to activations
            prototype_activations = self.distance_2_similarity(min_distances)
            #print(f"[DEBUG] Prototype Activations Shape: {prototype_activations.shape}")  # Expected: (batch, num_prototypes)

            # Compute logits: (batch, num_classes)
            logits = self.classifier(prototype_activations)
            #print(f"[DEBUG] Logits Shape: {logits.shape}")  # Expected: (batch, num_classes)

            return logits, min_distances, prototype_activations

        elif self.latent_space_type == "arc":
            activations, marginless_activations = self.cos_activation(features, prototypes_of_wrong_class)
            #print(f"[DEBUG] Cosine Activation Shape: {activations.shape}")  # Expected: (batch_size, num_prototypes, num_time_windows)

            # Pool over time (reduce num_time_windows)
            prototype_activations = torch.mean(activations, dim=-1)  # (batch_size, num_prototypes)
            marginless_prototype_activations = torch.mean(marginless_activations, dim=-1)

            # Compute logits: (batch_size, num_classes) 
            logits = self.classifier(prototype_activations)  
            marginless_logits = self.classifier(marginless_prototype_activations)

            #print(f"[DEBUG] Logits Shape: {logits.shape}")  # Expected: (batch_size, num_classes)

            return logits, marginless_logits, prototype_activations

    def push_forward(self, x):
        """Extracts feature representations and similarity scores for prototype pushing."""
        
        features = self.conv_features(x)  # Extract feature representations

        if self.latent_space_type == "l2":
            distances = self._l2_convolution(features)  # Compute L2 distances
            activations = self.distance_2_similarity(distances)  # Convert distances to activations
            return features, activations  # (batch_size, feature_dim), (batch_size, num_prototypes)

        elif self.latent_space_type == "arc":
            activations, marginless_activations = self.cos_activation(features)  # Get both activations
            return features, activations  # Use actual activations (not marginless)

        else:
            raise ValueError(f"Unsupported latent space type: {self.latent_space_type}")

    def _l2_convolution(self, x):
        """
        Computes Euclidean (L2) distance between feature vectors and prototype vectors.
        
        - x: (batch, feature_dim, 1, time) → Need to reshape to (batch, feature_dim, time)
        - self.prototype_vectors: (num_prototypes, feature_dim, proto_time_len)

        Returns:
        - distances: (batch, num_prototypes, time - proto_time_len + 1)
        """

        # Ensure x is (batch, feature_dim, time)
        x = x.squeeze(2)  # Remove singleton dimension → (batch, feature_dim, time)
        batch_size, feature_dim, time_dim = x.shape

        num_windows = time_dim - self.proto_time_len + 1  # Compute sliding window count

        # Ensure prototype vectors are also 3D: (num_prototypes, feature_dim, proto_time_len)
        prototypes = self.prototype_vectors.unsqueeze(0)  # Add batch dim → (1, num_prototypes, feature_dim, proto_time_len)

        # Initialize distance tensor
        distances = torch.zeros((batch_size, self.num_prototypes, num_windows), device=x.device)

        # Compute L2 distances across time windows
        for i in range(num_windows):
            segment = x[:, :, i : i + self.proto_time_len]  # (batch, feature_dim, proto_time_len)
            
            # Reshape for cdist: (batch, 1, feature_dim * proto_time_len) vs (1, num_prototypes, feature_dim * proto_time_len)
            segment = segment.reshape(batch_size, feature_dim * self.proto_time_len)
            prototypes = self.prototype_vectors.reshape(self.num_prototypes, feature_dim * self.proto_time_len)

            # Compute pairwise L2 distances
            distances[:, :, i] = torch.cdist(segment.unsqueeze(1), prototypes.unsqueeze(0), p=2).squeeze(1)

        return distances  # Shape: (batch, num_prototypes, time_windows)

    def cos_activation(self, x, prototypes_of_wrong_class=None):
        """
        Computes scaled cosine similarity (not bounded to [-1, 1]) between inputs and prototypes.
        Supports both partial and global prototypes.
        """
        eps = self.epsilon
        input_vector_length = self.input_vector_length  # e.g., 64, affects scale
        normalizing_factor = (self.prototype_shape[-2] * self.prototype_shape[-1]) ** 0.5  # sqrt(F × T)

        batch_size, feature_dim, _, time_dim = x.shape
        proto_len = self.proto_time_len
        num_windows = time_dim - proto_len + 1

        activations = torch.zeros((batch_size, self.num_prototypes, num_windows), device=x.device)

        # Prepare normalized prototypes
        prototypes = self.prototype_vectors.squeeze(2)  # (P, F, T)
        proto_norm = torch.sqrt(torch.sum(prototypes ** 2, dim=(1, 2), keepdim=True) + eps)
        prototypes_scaled = prototypes / proto_norm

        prototypes_scaled = prototypes_scaled / normalizing_factor 

        for i in range(num_windows):
            segment = x[:, :, 0, i : i + proto_len]  # (B, F, T)
            seg_norm = torch.sqrt(torch.sum(segment ** 2, dim=(1, 2), keepdim=True) + eps)
            segment_scaled = input_vector_length * segment / seg_norm
            
            segment_scaled = segment_scaled / normalizing_factor 

            # Cosine-like similarity (dot product of normalized & scaled vectors)
            # Reshape: (B, F*T) x (P, F*T)^T → (B, P)
            seg_flat = segment_scaled.view(batch_size, -1)
            proto_flat = prototypes_scaled.view(self.num_prototypes, -1)
            activations[:, :, i] = torch.matmul(seg_flat, proto_flat.T)

        return activations, activations  # no margin adjustment

    def get_prototype_orthogonalities(self):
        """
        Computes orthogonality loss for 2D prototypes.
        """
        # Flatten and normalize prototypes: shape → (num_prototypes, F × T)
        P_flat = self.prototype_vectors.view(self.num_prototypes, -1)  # (P, D)
        P_norm = F.normalize(P_flat, p=2, dim=1)  # L2 normalize each prototype

        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(P_norm, P_norm.T)  # (P, P)

        # Subtract identity to ignore self-similarity
        identity = torch.eye(self.num_prototypes, device=self.prototype_vectors.device)
        off_diag_similarity = similarity_matrix - identity

        # Orthogonality loss = sum of squared off-diagonal cosine similarities
        orthogonality_loss = (off_diag_similarity ** 2).sum()

        return orthogonality_loss


    def prune_prototypes(self, prototypes_to_prune):
        """
        Removes specified prototypes from the model.

        Args:
            prototypes_to_prune: List of prototype indices to remove.
        """
        # Ensure valid prototype indices
        if not prototypes_to_prune:
            print("No prototypes to prune.")
            return

        print(f"Pruning prototypes: {prototypes_to_prune}")

        # Determine which prototypes to keep
        prototypes_to_keep = sorted(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        # Update prototype vectors
        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep], requires_grad=True)

        # Update prototype shape & count
        self.num_prototypes = len(prototypes_to_keep)
        self.prototype_shape = list(self.prototype_vectors.size())

        # Create a new classifier with updated input size
        old_weights = self.classifier.weight.data[:, prototypes_to_keep].clone()  # Keep only valid weights
        self.classifier = nn.Linear(self.num_prototypes, self.num_classes, bias=False).to(self.prototype_vectors.device)

        # Copy back the old weights safely
        with torch.no_grad():
            self.classifier.weight.copy_(old_weights)

        # Update `self.ones` 
        if hasattr(self, "ones"):
            self.ones = nn.Parameter(self.ones.data[prototypes_to_keep], requires_grad=False)

        # Update prototype class identity matrix
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

        print(f"New prototype count: {self.num_prototypes}")

    def __repr__(self):
        """String representation of the model for debugging and inspection."""
        rep = (
            "ProtoECGNet2D(\n"
            "\tfeatures: {},\n"
            "\tprototype_shape: {},\n"
            "\tnum_classes: {},\n"
            "\tepsilon: {}\n"
            ")"
        )

        return rep.format(self.feature_extractor, self.prototype_shape, self.num_classes, self.epsilon)

