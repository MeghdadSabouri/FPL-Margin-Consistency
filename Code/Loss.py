import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedQuantumLoss(nn.Module):
    def __init__(self, feat_dim=2048, num_classes=4, ce_weight=0.7, contrastive_weight=0.1, quantum_weight=0.2, temperature=1.0, alpha=0.5, beta=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.ce_weight, self.contrastive_weight, self.quantum_weight = ce_weight, contrastive_weight, quantum_weight
        self.temperature, self.alpha, self.beta = temperature, alpha, beta
        self.register_parameter('W_h', nn.Parameter(torch.ones(feat_dim)))
        self.register_parameter('sigma', nn.Parameter(torch.eye(feat_dim)))
        self.register_buffer('class_weights', torch.tensor([1.0, 1.0, 1.0, 1.0]))

    def compute_structure_tensor(self, x):
        grad_x = torch.gradient(x, dim=1)[0]
        return torch.bmm(grad_x.unsqueeze(-1), grad_x.unsqueeze(1))

    def generate_perturbation(self, x):
        S = self.compute_structure_tensor(x)
        structure_term = self.alpha * torch.matmul(S, x.unsqueeze(-1)).squeeze(-1)
        sigma_regularized = self.sigma + 1e-6 * torch.eye(self.feat_dim, device=x.device)
        try:
            chol_sigma = torch.linalg.cholesky(sigma_regularized)
            random_term = self.beta * torch.matmul(torch.randn_like(x), chol_sigma)
        except RuntimeError:
            random_term = self.beta * torch.randn_like(x)
        return structure_term + random_term

    def quantum_state(self, x, perturbed=False):
        x_norm = F.normalize(x, dim=1, eps=1e-8)
        if perturbed:
            x_norm = x_norm + self.generate_perturbation(x_norm)
            x_norm = F.normalize(x_norm, dim=1, eps=1e-8)
        return torch.exp(1j * torch.pi * F.relu(self.W_h) * x_norm)

    def quantum_fidelity(self, x1, x2):
        psi_1 = self.quantum_state(x1)
        psi_2 = self.quantum_state(x2, perturbed=True)
        fidelity = torch.abs(torch.sum(torch.conj(psi_1) * psi_2, dim=1))
        return torch.clamp(fidelity * 0.5 * (1 + F.cosine_similarity(x1, x2, dim=1)), 0, 1)

    def compute_quantum_loss(self, features, labels):
        batch_size = features.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device)
            
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        loss = 0
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    fidelity = self.quantum_fidelity(features[i:i+1], features[j:j+1])
                    loss += (1 - fidelity) if label_matrix[i, j] else fidelity
        return loss / (batch_size * (batch_size - 1))

    def forward(self, features, outputs, labels):
        ce_loss = F.cross_entropy(outputs, labels, weight=self.class_weights.to(features.device))
        quantum_loss = self.compute_quantum_loss(features, labels)
        return self.ce_weight * ce_loss + self.quantum_weight * quantum_loss
