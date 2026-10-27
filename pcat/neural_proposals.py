"""Optional neural density proposals for transdimensional element moves."""

from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp, ndtr
from scipy.stats import truncnorm
from tdpy.verbosity import print


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise ImportError(
            "NeuralElementProposal requires PyTorch. Install it with "
            "`python -m pip install 'pcat[neural]'`."
        ) from error
    return torch, nn


def _make_network(condition_dimension, element_dimension, hidden_sizes, number_components):
    torch, nn = _require_torch()
    layers = []
    input_dimension = condition_dimension
    for hidden_size in hidden_sizes:
        layers.extend((nn.Linear(input_dimension, hidden_size), nn.SiLU()))
        input_dimension = hidden_size
    output_dimension = number_components * (1 + 2 * element_dimension)
    layers.append(nn.Linear(input_dimension, output_dimension))
    return nn.Sequential(*layers)

class _StateProposalCallbacks:
    def __init__(self, proposal, retr_condition):
        self.proposal = proposal
        self.retr_condition = retr_condition

    def draw(self, gdat, context):
        return self.proposal.sample(self.retr_condition(gdat, context))

    def log_prob(self, gdat, unit_element, context):
        return self.proposal.log_prob(self.retr_condition(gdat, context), unit_element)

    def callbacks(self):
        return {
            "retr_drawpropelem_state": self.draw,
            "retr_lpdfpropelem_state": self.log_prob,
        }


class NeuralElementProposal:
    """Conditional mixture density on PCAT's unit-prior element coordinates.

    The density is a neural mixture of factorized truncated Gaussians with a
    uniform component. It implements PCAT's ``retr_drawpropelem`` and
    ``retr_lpdfpropelem`` callback signatures using a feature vector stored on
    the run configuration as ``neural_proposal_features``.
    """

    def __init__(
        self,
        network,
        condition_mean,
        condition_scale,
        element_dimension,
        number_components,
        hidden_sizes,
        uniform_fraction,
        minimum_scale,
        feature_attribute="neural_proposal_features",
        loss_history=None,
    ):
        self.network = network.eval()
        self.condition_mean = np.asarray(condition_mean, dtype=np.float32)
        self.condition_scale = np.asarray(condition_scale, dtype=np.float32)
        self.condition_dimension = self.condition_mean.size
        self.element_dimension = int(element_dimension)
        self.number_components = int(number_components)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.uniform_fraction = float(uniform_fraction)
        self.minimum_scale = float(minimum_scale)
        self.feature_attribute = feature_attribute
        self.loss_history = list(loss_history or [])

    @classmethod
    def fit(
        cls,
        conditions,
        unit_elements,
        *,
        number_components=8,
        hidden_sizes=(64, 64),
        epochs=300,
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=1e-5,
        uniform_fraction=0.05,
        minimum_scale=1e-3,
        seed=0,
        feature_attribute="neural_proposal_features",
    ):
        """Fit the conditional density to simulated feature/element pairs."""
        torch, _ = _require_torch()
        conditions = np.asarray(conditions, dtype=np.float32)
        unit_elements = np.asarray(unit_elements, dtype=np.float32)
        hidden_sizes = tuple(int(size) for size in hidden_sizes)

        if conditions.ndim != 2 or conditions.shape[0] < 2 or conditions.shape[1] < 1:
            raise ValueError("conditions must be a 2D array with at least two rows and one feature.")
        if unit_elements.ndim != 2 or unit_elements.shape[0] != conditions.shape[0]:
            raise ValueError("unit_elements must be a 2D array with one row per condition.")
        if unit_elements.shape[1] < 1:
            raise ValueError("unit_elements must contain at least one element parameter.")
        if not np.isfinite(conditions).all() or not np.isfinite(unit_elements).all():
            raise ValueError("training arrays must contain only finite values.")
        if np.any((unit_elements < 0.) | (unit_elements > 1.)):
            raise ValueError("unit_elements must lie in the unit cube [0, 1].")
        if number_components < 1 or any(size < 1 for size in hidden_sizes):
            raise ValueError("number_components and hidden_sizes must be positive.")
        if epochs < 1 or batch_size < 1 or learning_rate <= 0. or weight_decay < 0.:
            raise ValueError("training settings must be positive, except weight_decay may be zero.")
        if not 0. < uniform_fraction < 1.:
            raise ValueError("uniform_fraction must be strictly between zero and one.")
        if not 0. < minimum_scale < 0.5:
            raise ValueError("minimum_scale must be between zero and 0.5.")

        condition_mean = conditions.mean(axis=0)
        condition_scale = conditions.std(axis=0)
        condition_scale[condition_scale == 0.] = 1.
        normalized_conditions = (conditions - condition_mean) / condition_scale
        training_conditions = torch.as_tensor(normalized_conditions)
        training_elements = torch.as_tensor(unit_elements)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            network = _make_network(
                conditions.shape[1], unit_elements.shape[1], hidden_sizes, number_components
            )

        proposal = cls(
            network=network,
            condition_mean=condition_mean,
            condition_scale=condition_scale,
            element_dimension=unit_elements.shape[1],
            number_components=number_components,
            hidden_sizes=hidden_sizes,
            uniform_fraction=uniform_fraction,
            minimum_scale=minimum_scale,
            feature_attribute=feature_attribute,
        )
        optimizer = torch.optim.AdamW(
            proposal.network.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        random = torch.Generator().manual_seed(int(seed))
        proposal.loss_history = []

        for _ in range(int(epochs)):
            permutation = torch.randperm(conditions.shape[0], generator=random)
            epoch_losses = []
            proposal.network.train()
            for indices in permutation.split(int(batch_size)):
                optimizer.zero_grad(set_to_none=True)
                log_density = proposal._log_prob_tensor(
                    training_conditions[indices], training_elements[indices]
                )
                loss = -log_density.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("neural proposal training produced a non-finite loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(proposal.network.parameters(), max_norm=10.)
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            proposal.loss_history.append(float(np.mean(epoch_losses)))

        proposal.network.eval()
        return proposal

    def _log_prob_tensor(self, conditions, unit_elements):
        torch, _ = _require_torch()
        output = self.network(conditions)
        mixture_logits, means_raw, scales_raw = torch.split(
            output,
            [self.number_components,
             self.number_components * self.element_dimension,
             self.number_components * self.element_dimension],
            dim=-1,
        )
        means = torch.sigmoid(means_raw.reshape(-1, self.number_components, self.element_dimension))
        scales = self.minimum_scale + (0.5 - self.minimum_scale) * torch.sigmoid(
            scales_raw.reshape(-1, self.number_components, self.element_dimension)
        )
        values = unit_elements[:, None, :]
        standardized = (values - means) / scales
        log_normalization = torch.log(
            torch.special.ndtr((1. - means) / scales)
            - torch.special.ndtr(-means / scales)
        )
        log_component = (
            -0.5 * standardized.square()
            - torch.log(scales)
            - 0.5 * np.log(2. * np.pi)
            - log_normalization
        ).sum(dim=-1)
        log_neural_density = torch.logsumexp(
            torch.log_softmax(mixture_logits, dim=-1) + log_component, dim=-1
        )
        log_uniform = torch.full_like(log_neural_density, np.log(self.uniform_fraction))
        return torch.logaddexp(
            log_neural_density + np.log1p(-self.uniform_fraction), log_uniform
        )

    def _condition_tensor(self, conditions):
        torch, _ = _require_torch()
        conditions = np.asarray(conditions, dtype=np.float32)
        if conditions.shape != (self.condition_dimension,) or not np.isfinite(conditions).all():
            raise ValueError(
                f"proposal features must be a finite vector of length {self.condition_dimension}."
            )
        normalized = (conditions - self.condition_mean) / self.condition_scale
        return torch.as_tensor(normalized[None, :])

    def log_prob(self, conditions, unit_element):
        """Return the normalized log density with respect to unit-cube volume."""
        return float(self.log_prob_batch(conditions, unit_element)[0])

    def log_prob_batch(self, conditions, unit_elements):
        """Return log densities for one or more unit-cube element vectors."""
        conditions = np.asarray(conditions, dtype=np.float32)
        unit_elements = np.asarray(unit_elements, dtype=np.float32)
        if unit_elements.ndim == 1:
            unit_elements = unit_elements[None, :]
        if unit_elements.ndim != 2 or unit_elements.shape[1] != self.element_dimension:
            raise ValueError(
                f"unit_elements must have shape (n, {self.element_dimension})."
            )
        shared_condition = conditions.ndim == 1
        if shared_condition:
            conditions = np.broadcast_to(conditions, (unit_elements.shape[0], conditions.size))
        if conditions.shape != (unit_elements.shape[0], self.condition_dimension):
            raise ValueError(
                f"conditions must have shape ({self.condition_dimension},) or "
                f"({unit_elements.shape[0]}, {self.condition_dimension})."
            )
        if not np.isfinite(conditions).all():
            raise ValueError("proposal features must contain only finite values.")
        valid = np.isfinite(unit_elements).all(axis=1)
        valid &= np.all((unit_elements >= 0.) & (unit_elements <= 1.), axis=1)
        result = np.full(unit_elements.shape[0], -np.inf, dtype=float)
        if not valid.any():
            return result

        torch, _ = _require_torch()
        with torch.no_grad():
            condition_rows = conditions[:1] if shared_condition else conditions[valid]
            normalized_conditions = (condition_rows - self.condition_mean) / self.condition_scale
            condition_tensor = torch.as_tensor(normalized_conditions)
            element_tensor = torch.as_tensor(unit_elements[valid])
            result[valid] = self._log_prob_tensor(condition_tensor, element_tensor).numpy()
        return result

    def sample(self, conditions):
        """Draw one element in unit-prior coordinates."""
        if np.random.random() < self.uniform_fraction:
            return np.random.random(self.element_dimension)

        torch, _ = _require_torch()
        with torch.no_grad():
            output = self.network(self._condition_tensor(conditions))[0].numpy()
        split = self.number_components
        weights = np.exp(output[:split] - logsumexp(output[:split]))
        means = expit(output[split:split + split * self.element_dimension])
        means = means.reshape(self.number_components, self.element_dimension)
        scales_raw = output[split + split * self.element_dimension:]
        scales = self.minimum_scale + (0.5 - self.minimum_scale) * expit(scales_raw)
        scales = scales.reshape(self.number_components, self.element_dimension)
        component = np.random.choice(self.number_components, p=weights)
        lower = -means[component] / scales[component]
        upper = (1. - means[component]) / scales[component]
        return np.asarray(
            truncnorm.rvs(lower, upper, loc=means[component], scale=scales[component]),
            dtype=float,
        )

    def draw_from_state(self, gdat):
        """PCAT proposal-draw callback using configured neural features."""
        features = getattr(gdat, self.feature_attribute, None)
        if features is None:
            raise ValueError(f"run configuration is missing {self.feature_attribute!r}.")
        return self.sample(features)

    def log_prob_from_state(self, gdat, unit_element):
        """PCAT proposal-density callback using configured neural features."""
        features = getattr(gdat, self.feature_attribute, None)
        if features is None:
            raise ValueError(f"run configuration is missing {self.feature_attribute!r}.")
        return self.log_prob(features, unit_element)

    def callbacks(self):
        """Return keyword arguments for PCAT's proposal callback interface."""
        return {
            "retr_drawpropelem": self.draw_from_state,
            "retr_lpdfpropelem": self.log_prob_from_state,
        }

    def state_callbacks(self, retr_condition):
        """Return callbacks conditioned on data and an immutable catalog context.

        ``retr_condition(gdat, context)`` must return the same feature vector
        representation used to fit this proposal. Training conditions must be
        constructed from simulated catalog states in the same way.
        """
        if not callable(retr_condition):
            raise TypeError("retr_condition must be callable.")
        return _StateProposalCallbacks(self, retr_condition).callbacks()

    def save(self, path):
        """Save model weights and density configuration for later sampling."""
        torch, _ = _require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing to {path}...")
        torch.save(
            {
                "condition_mean": self.condition_mean.tolist(),
                "condition_scale": self.condition_scale.tolist(),
                "condition_dimension": self.condition_dimension,
                "element_dimension": self.element_dimension,
                "number_components": self.number_components,
                "hidden_sizes": self.hidden_sizes,
                "uniform_fraction": self.uniform_fraction,
                "minimum_scale": self.minimum_scale,
                "feature_attribute": self.feature_attribute,
                "state_dict": self.network.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path):
        """Load a saved proposal without unpickling arbitrary Python objects."""
        torch, _ = _require_torch()
        path = Path(path)
        print(f"Reading from {path}...")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        network = _make_network(
            checkpoint["condition_dimension"],
            checkpoint["element_dimension"],
            checkpoint["hidden_sizes"],
            checkpoint["number_components"],
        )
        network.load_state_dict(checkpoint["state_dict"])
        return cls(
            network=network,
            condition_mean=checkpoint["condition_mean"],
            condition_scale=checkpoint["condition_scale"],
            element_dimension=checkpoint["element_dimension"],
            number_components=checkpoint["number_components"],
            hidden_sizes=checkpoint["hidden_sizes"],
            uniform_fraction=checkpoint["uniform_fraction"],
            minimum_scale=checkpoint["minimum_scale"],
            feature_attribute=checkpoint["feature_attribute"],
        )


__all__ = ["NeuralElementProposal"]