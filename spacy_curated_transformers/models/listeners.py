from typing import Any, Callable, Iterable, List, Optional, Tuple

from spacy import Errors as SpacyErrors
from spacy.tokens import Doc
from thinc.api import Model, deserialize_attr, serialize_attr
from thinc.types import Floats2d, Ragged

from .output import TransformerModelOutput
from .pooling import with_ragged_last_layer, with_ragged_layers
from .types import (
    PoolingModelT,
    ScalarWeightModelT,
    ScalarWeightOutT,
    TransformerListenerModelT,
    TransformerModelT,
    WithRaggedLastLayerModelT,
    WithRaggedLayersModelT,
    WrappedTransformerAndListenerBackpropT,
    WrappedTransformerAndListenerInT,
    WrappedTransformerAndListenerModelT,
    WrappedTransformerAndListenerOutT,
)


def build_transformer_layers_listener_v1(
    layers: int,
    width: int,
    pooling: PoolingModelT,
    upstream: str = "*",
    grad_factor: float = 1.0,
) -> TransformerListenerModelT:
    """Construct a listener layer that communicates with one or more upstream Transformer
    components. This layer extracts the output of all transformer layers and performs
    pooling over the individual pieces of each Doc token, returning their corresponding
    representations.

    layers (int):
        The the number of layers produced by the upstream transformer component,
        excluding the embedding layer.
    width (int):
        The width of the vectors produced by the upstream transformer component.
    pooling (PoolingModelT):
        Model that is used to perform pooling over the piece representations.
    upstream (str):
        A string to identify the 'upstream' Transformer component
        to communicate with. The upstream name should either be the wildcard
        string '*', or the name of the Transformer component.

        In almost all cases, the wildcard string will suffice as there'll only be one
        upstream Transformer component. But in certain situations, e.g: you have disjoint
        datasets for certain tasks, or you'd like to use a pre-trained pipeline but a
        downstream task requires its own token representations, you could end up with
        more than one Transformer component in the pipeline.
    grad_factor (float):
        Factor to multiply gradients with.
    """
    model: TransformerListenerModelT = Model(
        name="transformer_layers_listener",
        forward=transformer_layers_listener_forward,
        dims={"nO": width},
        layers=[with_ragged_layers(pooling)],
        attrs={
            "grad_factor": grad_factor,
            "layers": layers + 1,
        },
        refs={"pooling": pooling},
    )
    ListenerStateUtils.init_state(
        model, upstream_name=upstream, requires_all_layer_outputs=True
    )
    return model


def build_last_transformer_layer_listener_v1(
    width: int,
    pooling: PoolingModelT,
    upstream: str = "*",
    grad_factor: float = 1.0,
) -> TransformerListenerModelT:
    """Construct a listener layer that communicates with one or more upstream Transformer
    components. This layer extracts the output of the last transformer layer and performs
    pooling over the individual pieces of each Doc token, returning their corresponding
    representations.

    width (int):
        The width of the vectors produced by the upstream transformer component.
    pooling (Model):
        Model that is used to perform pooling over the piece representations.
    upstream (str):
        A string to identify the 'upstream' Transformer component
        to communicate with. The upstream name should either be the wildcard
        string '*', or the name of the Transformer component.

        In almost all cases, the wildcard string will suffice as there'll only be one
        upstream Transformer component. But in certain situations, e.g: you have disjoint
        datasets for certain tasks, or you'd like to use a pre-trained pipeline but a
        downstream task requires its own token representations, you could end up with
        more than one Transformer component in the pipeline.
    grad_factor (float):
        Factor to multiply gradients with.
    """
    model: TransformerListenerModelT = Model(
        name="last_transformer_layer_listener",
        forward=last_transformer_layer_listener_forward,
        dims={"nO": width},
        layers=[with_ragged_last_layer(pooling)],
        attrs={"grad_factor": grad_factor},
        refs={"pooling": pooling},
    )
    ListenerStateUtils.init_state(
        model, upstream_name=upstream, requires_all_layer_outputs=False
    )
    return model


def build_scalar_weighting_listener_v1(
    width: int,
    weighting: ScalarWeightModelT,
    pooling: PoolingModelT,
    upstream: str = "*",
    grad_factor: float = 1.0,
) -> TransformerListenerModelT:
    """Construct a listener layer that communicates with one or more upstream Transformer
    components. This layer calculates a weighted representation of all transformer layer
    outputs and performs pooling over the individual pieces of each Doc token, returning
    their corresponding representations.

    Requires its upstream Transformer components to return all layer outputs from
    their models.

    width (int):
        The width of the vectors produced by the upstream transformer component.
    weighting (Model):
        Model that is used to perform the weighting of the different layer outputs.
    pooling (Model):
        Model that is used to perform pooling over the piece representations.
    upstream (str):
        A string to identify the 'upstream' Transformer component
        to communicate with. The upstream name should either be the wildcard
        string '*', or the name of the Transformer component.

        In almost all cases, the wildcard string will suffice as there'll only be one
        upstream Transformer component. But in certain situations, e.g: you have disjoint
        datasets for certain tasks, or you'd like to use a pre-trained pipeline but a
        downstream task requires its own token representations, you could end up with
        more than one Transformer component in the pipeline.
    grad_factor (float):
        Factor to multiply gradients with.
    """
    model: TransformerListenerModelT = Model(
        name="scalar_weighting_listener",
        forward=scalar_weighting_listener_forward,
        dims={"nO": width},
        layers=[weighting, with_ragged_last_layer(pooling)],
        attrs={
            "grad_factor": grad_factor,
        },
    )
    ListenerStateUtils.init_state(
        model, upstream_name=upstream, requires_all_layer_outputs=True
    )
    return model


# We need to store the listener state in the attributes dict of the listener model.
# This custom class is used to ensure that this state is not serialized to disk.
class _ListenerNonPersistentState:
    batch_id: Optional[int] = None
    outputs: Optional[TransformerModelOutput] = None
    backprop: Optional[Callable[[List[List[Ragged]], Tuple[int]], Any]] = None


@serialize_attr.register(_ListenerNonPersistentState)
def serialize_listener_non_persistent_state(
    _, value: _ListenerNonPersistentState, name: str, model
) -> bytes:
    return bytes()


@deserialize_attr.register(_ListenerNonPersistentState)
def deserialize_listener_non_persistent_state(
    _, value: bytes, name: str, model
) -> _ListenerNonPersistentState:
    return _ListenerNonPersistentState()


class ListenerStateUtils:
    """This class provides helper functions to manage the internal state of
    `Model`s that act as listeners of the `Transformer` pipe."""

    SENTINEL = "_TRANSFORMER_LISTENER"
    USE_DOC_ANNOTATIONS_FOR_PREDICTION = "_USE_DOC_ANNOTATIONS_FOR_PREDICTION"
    REQUIRES_ALL_LAYER_OUTPUTS = "_REQUIRES_ALL_LAYER_OUTPUTS"

    @classmethod
    def is_listener(cls, model: Model) -> bool:
        return cls.SENTINEL in model.attrs

    @classmethod
    def init_state(
        cls,
        listener: TransformerListenerModelT,
        *,
        upstream_name: str = "*",
        requires_all_layer_outputs: bool
    ):
        """Initialize the listener model."""
        listener.attrs["_upstream_name"] = upstream_name
        listener.attrs[cls.SENTINEL] = True
        listener.attrs[cls.USE_DOC_ANNOTATIONS_FOR_PREDICTION] = True
        listener.attrs[cls.REQUIRES_ALL_LAYER_OUTPUTS] = requires_all_layer_outputs
        listener.attrs["_state"] = _ListenerNonPersistentState()

    @classmethod
    def use_doc_annotations_for_prediction(
        cls, listener: TransformerListenerModelT
    ) -> bool:
        """If True, the listener will perform its operations on the transformer output
        annotations stored on the Doc objects. Otherwise, it will perform its operations
        on the outputs that were stored directly in it using `TransformerListener.receive`.
        """
        return listener.attrs[cls.USE_DOC_ANNOTATIONS_FOR_PREDICTION]

    @classmethod
    def set_use_doc_annotations_for_prediction(
        cls, listener: TransformerListenerModelT, new_value: bool
    ):
        listener.attrs[cls.USE_DOC_ANNOTATIONS_FOR_PREDICTION] = new_value

    @classmethod
    def requires_all_layer_outputs(cls, model: Model) -> bool:
        return model.attrs[cls.REQUIRES_ALL_LAYER_OUTPUTS]

    @classmethod
    def get_upstream_name(cls, listener: TransformerListenerModelT) -> str:
        return listener.attrs["_upstream_name"]

    @staticmethod
    def get_batch_id(listener: TransformerListenerModelT) -> Optional[int]:
        state: _ListenerNonPersistentState = listener.attrs["_state"]
        return state.batch_id

    @staticmethod
    def get_output(
        listener: TransformerListenerModelT,
    ) -> Optional[TransformerModelOutput]:
        state: _ListenerNonPersistentState = listener.attrs["_state"]
        return state.outputs

    @staticmethod
    def get_backprop(
        listener: TransformerListenerModelT,
    ) -> Optional[Callable[[List[List[Ragged]], Tuple[int]], Any]]:
        state: _ListenerNonPersistentState = listener.attrs["_state"]
        return state.backprop

    @staticmethod
    def calculate_batch_id(inputs: Iterable[Doc]) -> int:
        """Calculate a content-sensitive hash of the batch of documents, to check
        whether the next batch of documents is unexpected.
        """
        return sum(sum(token.orth for token in doc) for doc in inputs)

    @staticmethod
    def receive(
        listener: TransformerListenerModelT,
        batch_id: int,
        outputs: TransformerModelOutput,
        backprop: Callable[[List[List[Ragged]], Tuple[int]], Any],
    ) -> None:
        """Store a batch of training predictions and a backprop callback in the listener.
        The predictions and callback are produced by the upstream Transformer component,
        and later will be used when the listener's component's model is called.
        """
        state: _ListenerNonPersistentState = listener.attrs["_state"]
        state.batch_id = batch_id
        state.outputs = outputs
        state.backprop = backprop

    @classmethod
    def verify_inputs(
        cls, listener: TransformerListenerModelT, inputs: Iterable[Doc]
    ) -> bool:
        """Check that the batch of Doc objects matches the ones we have a
        prediction for.
        """
        expected_batch_id = cls.get_batch_id(listener)
        outputs = cls.get_output(listener)

        if expected_batch_id is None and outputs is None:
            raise ValueError(SpacyErrors.E954)
        else:
            batch_id = cls.calculate_batch_id(inputs)
            if expected_batch_id != batch_id:
                raise ValueError(
                    SpacyErrors.E953.format(id1=expected_batch_id, id2=batch_id)
                )
            else:
                return True

    @staticmethod
    def clear_state(
        listener: TransformerListenerModelT,
    ):
        state: _ListenerNonPersistentState = listener.attrs["_state"]
        state.batch_id = None
        state.outputs = None
        state.backprop = None


def transformer_layers_listener_forward(
    model: TransformerListenerModelT, docs: Iterable[Doc], is_train: bool
) -> Tuple[List[List[Floats2d]], Callable[[Any], Any]]:
    pooling: WithRaggedLayersModelT = model.layers[0]
    grad_factor: float = model.attrs["grad_factor"]
    n_layers: int = model.attrs["layers"]

    _outputs = ListenerStateUtils.get_output(model)
    _backprop = ListenerStateUtils.get_backprop(model)

    if is_train:
        assert _outputs is not None
        if _outputs.last_layer_only:
            raise ValueError
        (
            "`TransformerLayersListener` requires the upstream transformer pipe to output "
            "all hidden layer outputs. This can be enabled by setting the pipe's "
            "`all_layer_outputs` parameter to `True` in the pipeline config"
        )

        ListenerStateUtils.verify_inputs(model, docs)

        Y, backprop_pooling = pooling(_outputs.all_outputs, is_train)

        def backprop(dY):
            dX = backprop_pooling(dY)

            if grad_factor != 1.0:
                for dX_doc in dX:
                    for dX_layer in dX_doc:
                        dX_layer.data *= grad_factor

            outputs_to_backprop = tuple(i for i in range(0, _outputs.num_outputs))
            dX = _backprop(dX, outputs_to_backprop=outputs_to_backprop)

            ListenerStateUtils.clear_state(model)
            return dX

        return Y, backprop

    else:
        if ListenerStateUtils.use_doc_annotations_for_prediction(model):
            width = model.get_dim("nO")

            no_trf_data = [doc._.trf_data is None for doc in docs]
            if any(no_trf_data):
                assert all(no_trf_data)
                return [
                    [model.ops.alloc2f(len(doc), width) for _ in range(n_layers)]
                    for doc in docs
                ], lambda dY: []

            if any(doc._.trf_data.last_layer_only for doc in docs):
                raise ValueError(
                    "`TransformerLayersListener` requires the upstream transformer pipe to output "
                    "all hidden layer outputs. This can be enabled by setting the pipe's "
                    "`all_layer_outputs` parameter to `True` in the pipeline config"
                )

            return pooling.predict(docs), lambda dY: []
        else:
            assert _outputs is not None
            ListenerStateUtils.verify_inputs(model, docs)
            outputs: Tuple[List[List[Floats2d]], Callable[[Any], Any]] = (
                pooling.predict(_outputs.all_outputs),
                lambda dY: [],
            )
            ListenerStateUtils.clear_state(model)
            return outputs


def last_transformer_layer_listener_forward(
    model: TransformerListenerModelT, docs: Iterable[Doc], is_train: bool
) -> Tuple[List[Floats2d], Callable[[Any], Any]]:
    pooling: WithRaggedLastLayerModelT = model.layers[0]
    grad_factor: float = model.attrs["grad_factor"]

    _outputs = ListenerStateUtils.get_output(model)
    _backprop = ListenerStateUtils.get_backprop(model)

    if is_train:
        assert _outputs is not None
        ListenerStateUtils.verify_inputs(model, docs)

        Y, backprop_pooling = pooling(_outputs.last_hidden_layer_states, is_train)

        def backprop(dY):
            dX_pooling = backprop_pooling(dY)
            if grad_factor != 1.0:
                for dx in dX_pooling:
                    dx.data *= grad_factor
            dX = _backprop([[d] for d in dX_pooling], outputs_to_backprop=(-1,))
            ListenerStateUtils.clear_state(model)

            return dX

        return Y, backprop
    else:
        if ListenerStateUtils.use_doc_annotations_for_prediction(model):
            width = model.get_dim("nO")

            no_trf_data = [doc._.trf_data is None for doc in docs]
            if any(no_trf_data):
                assert all(no_trf_data)
                return [
                    model.ops.alloc2f(len(doc), width) for doc in docs
                ], lambda dY: []

            return pooling.predict(docs), lambda dY: []
        else:
            assert _outputs is not None
            ListenerStateUtils.verify_inputs(model, docs)
            outputs: Tuple[List[Floats2d], Callable[[Any], Any]] = (
                pooling.predict(_outputs.last_hidden_layer_states),
                lambda dY: [],
            )
            ListenerStateUtils.clear_state(model)
            return outputs


def scalar_weighting_listener_forward(
    model: TransformerListenerModelT, docs: Iterable[Doc], is_train: bool
) -> Tuple[List[Floats2d], Callable[[Any], Any]]:
    weighting: ScalarWeightModelT = model.layers[0]
    pooling: WithRaggedLastLayerModelT = model.layers[1]
    grad_factor: float = model.attrs["grad_factor"]

    _outputs = ListenerStateUtils.get_output(model)
    _backprop = ListenerStateUtils.get_backprop(model)

    if is_train:
        assert _outputs is not None
        if _outputs.last_layer_only:
            raise ValueError(
                "`ScalarWeightingListener` requires the upstream transformer pipe to output "
                "all hidden layer outputs. This can be enabled by setting the pipe's "
                "`all_layer_outputs` parameter to `True` in the pipeline config"
            )

        ListenerStateUtils.verify_inputs(model, docs)

        Y_weighting: ScalarWeightOutT = []
        weighting_inputs = _outputs.all_outputs
        outputs_to_backprop = tuple(i for i in range(_outputs.num_outputs))

        Y_weighting, backprop_weighting = weighting(weighting_inputs, is_train)
        Y, backprop_pooling = pooling(Y_weighting, is_train)

        def backprop(dYs):
            dX_pooling = backprop_pooling(dYs)
            dX_weighting = backprop_weighting(dX_pooling)

            if grad_factor != 1.0:
                for dx_inner in dX_weighting:
                    for dx in dx_inner:
                        dx.data *= grad_factor

            dX = _backprop(dX_weighting, outputs_to_backprop=outputs_to_backprop)
            ListenerStateUtils.clear_state(model)
            return dX

        return Y, backprop
    else:
        if ListenerStateUtils.use_doc_annotations_for_prediction(model):
            width = model.get_dim("nO")

            no_trf_data = [doc._.trf_data is None for doc in docs]
            if any(no_trf_data):
                assert all(no_trf_data)
                return [
                    model.ops.alloc2f(len(doc), width) for doc in docs
                ], lambda dY: []

            if any(doc._.trf_data.last_layer_only for doc in docs):
                raise ValueError(
                    "`ScalarWeightingListener` requires the upstream transformer pipe to output "
                    "all hidden layer outputs. This can be enabled by setting the pipe's "
                    "`all_layer_outputs` parameter to `True` in the pipeline config"
                )

            Y_weighting = weighting.predict(
                [doc._.trf_data.all_outputs for doc in docs]
            )
            Y = pooling.predict(Y_weighting)

            return Y, lambda dX: []
        else:
            assert _outputs is not None
            ListenerStateUtils.verify_inputs(model, docs)
            Y_weighting = weighting.predict(_outputs.all_outputs)
            outputs: Tuple[List[Floats2d], Callable[[Any], Any]] = (
                pooling.predict(Y_weighting),
                lambda dY: [],
            )
            ListenerStateUtils.clear_state(model)
            return outputs


class WrappedTransformerAndListener(WrappedTransformerAndListenerModelT):
    """Wraps a transformer model and a compatible listener. Exclusively used when replacing
    listeners of a shared transformer pipeline."""

    name = "wrapped_transformer_and_listener"

    def __init__(
        self,
        transformer: TransformerModelT,
        listener: TransformerListenerModelT,
        frozen: bool = False,
    ) -> None:
        """
        transformer (TransformerModelT):
            Transformer model to wrap.
        listener (TransformerListenerModelT):
            Listener model to wrap.
        frozen (bool):
            If the transformer is frozen.
        """
        Model.__init__(
            self,
            name=self.name,
            init=wrapped_transformer_and_listener_init,
            forward=wrapped_transformer_and_listener_forward,
            dims={"nO": listener.get_dim("nO")},
            layers=[transformer, listener],
        )

        # Ensure that the transformer returns the required outputs.
        transformer.attrs[
            "_all_layer_outputs"
        ] = ListenerStateUtils.requires_all_layer_outputs(listener)

        # Freeze the embedded transformer if the source pipe was frozen.
        transformer.attrs["_frozen"] = frozen

        # Remove the marker from the wrapped listener so that it doesn't
        # get registered to any upstream transformer pipes.
        assert ListenerStateUtils.SENTINEL in listener.attrs
        del listener.attrs[ListenerStateUtils.SENTINEL]

        # Ensure that the listener directly reads from the last batch of transformer
        # outputs stored on it during prediction. If we don't do this, the wrapped
        # listener will end up using the annotations of the last upstream transformer
        # pipe that set the  `trf_data` annotation on the Doc object.
        assert ListenerStateUtils.use_doc_annotations_for_prediction(listener)
        ListenerStateUtils.set_use_doc_annotations_for_prediction(listener, False)

    @property
    def frozen_transformer(self) -> bool:
        return self.layers[0].attrs["_frozen"]

    @frozen_transformer.setter
    def frozen_transformer(self, value: bool):
        self.layers[0].attrs["_frozen"] = value


def wrapped_transformer_and_listener_init(model: WrappedTransformerAndListener, X, Y):
    transformer: TransformerModelT = model.layers[0]
    listener: TransformerListenerModelT = model.layers[1]

    transformer.init(X=X, Y=Y)
    listener.init(X=X, Y=Y)


def wrapped_transformer_and_listener_forward(
    model: WrappedTransformerAndListener,
    docs: WrappedTransformerAndListenerInT,
    is_train: bool,
) -> Tuple[WrappedTransformerAndListenerOutT, WrappedTransformerAndListenerBackpropT]:
    transformer: TransformerModelT = model.layers[0]
    listener: TransformerListenerModelT = model.layers[1]
    frozen: bool = transformer.attrs["_frozen"]
    ops = model.ops

    # Follows the same process as `Transformer.update()`.
    if frozen or not is_train:
        outputs = transformer.predict(docs)
        bp_outputs = None
        d_outputs = None
    else:
        outputs, bp_outputs = transformer(docs, is_train)
        d_outputs = [
            [Ragged(ops.alloc_f(t2v.dataXd.shape), t2v.lengths) for t2v in doc_layers]
            for doc_layers in outputs.all_outputs
        ]

    def backprop(
        one_d_outputs: List[List[Ragged]], outputs_to_backprop: Tuple[int, ...]
    ) -> Any:
        nonlocal d_outputs
        nonlocal frozen

        if frozen:
            return []

        assert bp_outputs is not None
        assert d_outputs is not None
        for i in range(len(one_d_outputs)):
            for j in outputs_to_backprop:
                d_outputs[i][j].data += one_d_outputs[i][j].data

        d_docs = bp_outputs(d_outputs)
        return d_docs

    # Set the output directly on the listener so that it can use them for both
    # training/backprop and prediction.
    ListenerStateUtils.receive(
        listener,
        batch_id=ListenerStateUtils.calculate_batch_id(docs),
        outputs=outputs,
        backprop=backprop,
    )
    listener_outputs, bp_listener_outputs = listener(docs, is_train)
    return listener_outputs, bp_listener_outputs


def replace_listener_callback(
    copied_trf_model: TransformerModelT,
    trf_listener: TransformerListenerModelT,
    trf_pipe: Any,
):
    # To avoid cyclic imports.
    from ..pipeline.transformer import CuratedTransformer

    assert isinstance(trf_pipe, CuratedTransformer)
    assert ListenerStateUtils.is_listener(trf_listener)

    copied_trf_listener = trf_listener.copy()
    wrapper = WrappedTransformerAndListener(
        transformer=copied_trf_model,
        listener=copied_trf_listener,
        frozen=trf_pipe.frozen,
    )
    return wrapper


def replace_listener_cfg_callback(trf_model_cfg, trf_listener_model_cfg):
    result = trf_model_cfg.copy()

    trf_model_cfg_arch = trf_model_cfg["@architectures"].split(".")
    assert len(trf_model_cfg_arch) == 3 and trf_model_cfg_arch[-2].endswith(
        "Transformer"
    )
    trf_listener_model_cfg_arch = trf_listener_model_cfg["@architectures"].split(".")
    assert len(trf_listener_model_cfg_arch) == 3 and trf_listener_model_cfg_arch[
        -2
    ].endswith("Listener")

    # The transformer model entrypoint should have a `wrapped_listener` parameter
    # that we can use to append the listener's config.
    result["wrapped_listener"] = trf_listener_model_cfg
    return result
