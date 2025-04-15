import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from functools import reduce
import operator as op
from backbones import (
    resnet1d18, resnet1d34, resnet1d50, 
    resnet1d101, resnet1d152
)

def prototype_loss1d(logits, y_true, model, similarity_scores, class_weights, 
                   lam_clst, lam_sep, lam_spars, lam_div, lam_cnrst, use_contrastive=True):
    """
    Prototype-based loss function for ProtoECGNet1D.
    - Uses precomputed similarity scores from the model's forward pass.
    """

    # Ensure all tensors are on the same device
    device = model.prototype_vectors.device
    y_true = y_true.to(device).float()
    logits = logits.to(device)
    similarity_scores = similarity_scores.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)

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

    # Sparsity Loss (Encourage fewer prototype activations per input); unused in our paper
    spars_loss = torch.mean(torch.clamp(similarity_scores, min=0).sum(dim=1))

    # Orthogonality Loss (Encourage diverse prototypes)
    P = F.normalize(model.prototype_vectors, p=2, dim=1)  # shape: (P, D)
    identity_matrix = torch.eye(P.shape[0], device=device)
    div_loss = torch.norm(torch.mm(P, P.T) - identity_matrix, p="fro") ** 2
    div_loss = div_loss / (model.num_prototypes ** 2)  # Normalize by number of prototypes

    if use_contrastive:
        # Flatten and normalize prototypes
        prototypes = model.prototype_vectors  # shape: (P, D)
        prototypes_flat = prototypes.view(prototypes.shape[0], -1)
        prototypes_norm = F.normalize(prototypes_flat, p=2, dim=1)  # shape: (P, D)

        # Cosine similarity between all prototype pairs
        sim_matrix = torch.mm(prototypes_norm, prototypes_norm.T)  # shape: (P, P)
        sim_matrix = torch.clamp(sim_matrix, min=-1 + 1e-6, max=1 - 1e-6)

        # Co-occurrence mask
        P_assign = model.prototype_class_identity.float()  # (P, C)
        L_cooccur = model.label_cooccurrence.to(device)

        # Compute pairwise label co-occurrence score between prototype class assignments
        pos_mask = torch.matmul(torch.matmul(P_assign, L_cooccur), P_assign.T)  # (P, P)
        neg_mask = 1 - pos_mask

        # Normalize masks to avoid scale issues
        pos_norm = pos_mask / (pos_mask.sum() + 1e-6)
        neg_norm = neg_mask / (neg_mask.sum() + 1e-6)

        # Compute contrastive loss
        pos_loss = torch.sum(sim_matrix * pos_norm)
        neg_loss = torch.sum(sim_matrix * neg_norm)
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

class ProtoECGNet1D(nn.Module):
    def __init__(self, backbone="resnet1d18", num_classes=5, single_class_prototype_per_class=5, joint_prototypes_per_border=2, proto_dim=512, 
                prototype_activation_function="log", latent_space_type="l2", add_on_layers_type="linear", class_specific=False, last_layer_connection_weight=None, m=None, dropout=0, 
                custom_groups=True, label_set='all', pretrained_weights=None):
   
        super().__init__()
        self.joint_prototypes_per_border = joint_prototypes_per_border
        self.single_class_prototype_per_class = single_class_prototype_per_class
        self.num_classes = num_classes

        num_class_pairs = self.ncr(num_classes, 2)  # Compute number of unique class pairs
        self.num_prototypes = (self.single_class_prototype_per_class * self.num_classes) + (self.joint_prototypes_per_border * num_class_pairs)

        self.prototype_shape = (self.num_prototypes, proto_dim)
        self.topk_k = 1  # not used in our global prototype implementation

        self.prototype_activation_function = prototype_activation_function
        self.class_specific=class_specific # unused
        self.epsilon = 1e-4 
        self.m = m # unused
        self.custom_groups = custom_groups
        self.label_set = label_set
        self.relu_on_cos = False # unused
        self.input_vector_length = proto_dim ** 0.5
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
        print(f"Initializing ProtoECGNet1D with backbone {backbone}...")
        self.feature_extractor, backbone_only = self._get_backbone(backbone, num_classes, dropout, pretrained_weights)
        feature_dim = proto_dim


        #Even if loading state_dict() from full model checkpoint, have to initialize all components
        self.feature_extractor.fc = nn.Identity()  # Remove classifier head
        
        #print(f"[DEBUG] Extracted Feature Dim: {feature_dim}") 
        
        self.prototype_vectors = nn.Parameter(torch.randn(*self.prototype_shape), requires_grad=True)
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
                self.prototype_shape = (self.num_prototypes, proto_dim)

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
        if add_on_layers_type == "other": #bottleneck
            self.add_on_layers = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, feature_dim),
                nn.Sigmoid()
            )
        elif add_on_layers_type == "identity":
            self.add_on_layers = nn.Identity()  # No transformation applied
        elif add_on_layers_type == "linear":
            self.add_on_layers = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Sigmoid()
            )
    
    def conv_features(self, x):
        """Extract feature representations from input."""
        x = self.feature_extractor(x)
        #print(f"[DEBUG] Feature Extractor Raw Output Shape: {x.shape}")  # Should be (batch, feature_dim)

        if x.dim() == 3:  # If output has (batch, C, T), apply pooling
            x = torch.mean(x, dim=-1)  # Reduce across time dimension (batch, C, T) → (batch, C)
            #print(f"[DEBUG] Feature Extractor Shape After Global Pooling: {x.shape}")

        if not isinstance(self.add_on_layers, nn.Identity):
            x = self.add_on_layers(x)  
            #print(f"[DEBUG] After Add-On Layers Shape: {x.shape}")

        return x  # Ensure (batch, feature_dim)

    @staticmethod
    def ncr(n, r):
        """Computes n choose r (combinations)."""
        r = min(r, n - r)
        numer = reduce(op.mul, range(n, n - r, -1), 1)
        denom = reduce(op.mul, range(1, r + 1), 1)
        return numer // denom

    def prototype_distances(self, x, prototypes_of_wrong_class=None):
        """Computes prototype distances using L2 convolution or cosine activation."""
        conv_features = self.conv_features(x)
        if self.latent_space_type == "l2":
            return self._l2_convolution(conv_features)
        elif self.latent_space_type == "arc":
            return self.cos_activation(conv_features, prototypes_of_wrong_class)
        else:
            raise ValueError(f"Unsupported latent space type: {self.latent_space_type}")
        
    def distance_2_similarity(self, distances):
        """Converts prototype distances to similarity scores based on activation function."""
        if self.prototype_activation_function == "log":
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == "linear":
            return -distances
        else:
            return self.prototype_activation_function(distances)
    
    def _initialize_weights(self):
        """Initializes weights for the model"""
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv1d): 
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):  
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
        """Selects and initializes a 1D backbone, loading full model if available."""
        
        backbones = {
            "resnet1d18": resnet1d18, "resnet1d34": resnet1d34, "resnet1d50": resnet1d50,
            "resnet1d101": resnet1d101, "resnet1d152": resnet1d152
        }
        
        if backbone not in backbones:
            raise ValueError(f"Unsupported 1D backbone: {backbone}")

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
                print(f"Full ProtoECGNet1D checkpoint detected. Skipping backbone-only loading...")
                backbone_only = False
            else:
                # Only load feature extractor weights, keeping prototypes/classifier unchanged
                print(f"Loading backbone-only weights from {pretrained_weights} (Prototypes & Classifier Unchanged)...")
                model.load_state_dict(state_dict, strict=False)
                backbone_only = True 

        else:
            print("No pretrained weights provided")

        return model, backbone_only
    
    def forward(self, x, prototypes_of_wrong_class=None):
        """Forward pass with feature extraction, prototype similarity, and classification."""

        features = self.conv_features(x)  # Extract feature representations
        #print(f"[DEBUG] Feature Extractor Output Shape: {features.shape}")  # Expected: (batch, feature_dim)

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
            #print(f"[DEBUG] Cosine Activation Shape: {activations.shape}")  # Expected: (batch_size, num_prototypes)

            # No max pooling for global prototypes, just use activations as they are
            prototype_activations = activations  # (batch_size, num_prototypes)
            marginless_prototype_activations = marginless_activations  # (batch_size, num_prototypes)

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
        - x: (batch, feature_dim) extracted feature representations.
        - self.prototype_vectors: (num_prototypes, feature_dim)

        Returns:
        - distances: (batch, num_prototypes) Euclidean distance from each feature to each prototype.
        """

        #print(f"[DEBUG] Input to L2 Convolution: {x.shape}")

        # Compute L2 distance 
        distances = torch.cdist(x, self.prototype_vectors, p=2)  # (batch, num_prototypes)

        #print(f"[DEBUG] L2 Distances Shape: {distances.shape}")
        return distances  # Ensure (batch, num_prototypes) shape

    def cos_activation(self, x, prototypes_of_wrong_class=None):
        """
        Computes cosine similarity between input features and prototype vectors.
        - x: (batch, feature_dim)
        - self.prototype_vectors: (num_prototypes, feature_dim)

        Returns:
        - activations: (batch, num_prototypes) Cosine similarity scores.
        - marginless_activations: (batch, num_prototypes) Same as activations (for now).
        """

        #print(f"[DEBUG] Input to Cosine Activation: {x.shape}")

        # Normalize features and prototypes
        x_normalized = F.normalize(x, p=2, dim=1)  # (batch, feature_dim)
        prototypes_normalized = F.normalize(self.prototype_vectors, p=2, dim=1)  # (num_prototypes, feature_dim)

        # Compute cosine similarity (batch, num_prototypes)
        activations = torch.mm(x_normalized, prototypes_normalized.T)  

        #print(f"[DEBUG] Cosine Activations Shape: {activations.shape}")

        # Margin-based cosine distance (not used here; so we just assign it to activations)
        marginless_activations = activations  
        return activations, marginless_activations

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
            "ProtoECGNet1D(\n"
            "\tfeatures: {},\n"
            "\tprototype_shape: {},\n"
            "\tnum_classes: {},\n"
            "\tepsilon: {}\n"
            ")"
        )

        return rep.format(self.feature_extractor, self.prototype_shape, self.num_classes, self.epsilon)

