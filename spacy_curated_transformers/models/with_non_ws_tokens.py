from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple, cast

from spacy.tokens import Doc
from thinc.api import Model, Ragged

from ..tokenization.types import Tok2PiecesInT
from .types import (
    WsTokenAdapterBackpropT,
    WsTokenAdapterInT,
    WsTokenAdapterModelT,
    WsTokenAdapterOutT,
)


@dataclass
class TokenAlignment:
    # The piece offset of the token within the doc with whitespace tokens.
    ws_piece_offset: int

    # The piece offset of the token within the doc without whitespace tokens.
    # Whitespace tokens have None as their offset, since they are not
    # represented in piece sequences without whitespace.
    no_ws_piece_offset: Optional[int]

    # The length of this sequence in pieces.
    n_pieces: int

    @property
    def is_whitespace(self) -> bool:
        return self.no_ws_piece_offset is None


@dataclass
class Alignment:
    # Token alignments.
    tokens: List[TokenAlignment]

    # The length of the document in pieces with whitespace tokens.
    ws_n_pieces: int

    # The length of the document in pieces without whitespace tokens.
    no_ws_n_pieces: int

    @property
    def has_no_whitespace(self) -> bool:
        return self.ws_n_pieces == self.no_ws_n_pieces

    def __iter__(self) -> Iterator[TokenAlignment]:
        return iter(self.tokens)


def with_non_ws_tokens(
    layer: Model[Tok2PiecesInT, WsTokenAdapterOutT]
) -> WsTokenAdapterModelT:
    """Removes non-whitespace tokens from the input before
    passing it to the inner model."""
    return Model(
        "with_non_ws_tokens",
        with_non_ws_tokens_forward,
        init=with_non_ws_tokens_init,
        layers=[layer],
    )


def with_non_ws_tokens_forward(
    model: Model, X: WsTokenAdapterInT, is_train: bool
) -> Tuple[WsTokenAdapterOutT, WsTokenAdapterBackpropT]:
    inner: Model[Tok2PiecesInT, WsTokenAdapterOutT] = model.layers[0]
    # The transformer doesn't expect whitespace tokens, so at some stage
    # they need to be removed before we pass the IDs in, as encoded by
    # the tokenizer.
    # Previous implementations did this as a wrapper around the whole
    # process: we removed the whitespace token IDs, encoded with the
    # transformer, and then recalculated the output arrays, inserting
    # empty rows for the whitespace tokens.
    # This was expensive, and the output manipulation is actually unnecessary.
    # The pooling layers which consume these representations actually support
    # lengths 0, for tokens in the document that don't correspond to any
    # wordpieces. We therefore just need to have 0-length entries in
    # our ragged array.
    # Here's a reminder of how the ragged representation works.
    # So let's say we have:
    # "We went to Timbuktu"
    # And this is wordpieced like:
    # [["We"], ["went"], ["to"], ["Tim", "buktu"]]
    # We'll have a ragged with regions of lengths [1, 1, 1, 2]
    # -- it's basically a nested list with the arrays concatenated.
    # If we have a whitespace token, we won't have any wordpiece
    # corresponding to it. This means we'll end up with a 0-length
    # alignment. This is fine -- the pooling operations all handle
    # 0-length ragged entries.
    # Instead of all this recalculation, what we've done instead is
    # update the tokenizers to make sure they insert the 0-length
    # entry for the whitespace tokens.
    # This lets us make this operation a noop. We leave the function
    # here however, so that the models trained previously still load
    # correctly. We need to remove this in future when we retrain.
    Y, backprop = inner(X, is_train=is_train)
    return Y, backprop


def with_non_ws_tokens_init(
    model: WsTokenAdapterModelT,
    X: Optional[WsTokenAdapterInT] = None,
    Y: Optional[WsTokenAdapterInT] = None,
) -> None:
    # Deprecated
    model.layers[0].initialize(X=_filter_tokens(X)[0] if X is not None else None, Y=Y)


def _create_alignments(
    model: Model, output: WsTokenAdapterOutT, ws_counts: List[Counter]
) -> List[Alignment]:
    """Create an alignment between whitespace and non-whitespace sequences."""
    # Deprecated
    alignments = []
    for doc_output, doc_ws_counts in zip(output.all_outputs, ws_counts):
        doc_alignments = []
        no_ws_offset = 0
        ws_offset = 0
        pieces_lens = model.ops.to_numpy(doc_output[0].lengths).tolist()
        for idx, pieces_len in enumerate(pieces_lens):
            # Whitespace tokens that preceded a token.
            n_ws = doc_ws_counts[idx]
            for i in range(ws_offset, ws_offset + n_ws):
                doc_alignments.append(
                    TokenAlignment(
                        ws_piece_offset=i, no_ws_piece_offset=None, n_pieces=1
                    )
                )
            ws_offset += n_ws

            # The token itself.
            doc_alignments.append(
                TokenAlignment(
                    n_pieces=pieces_len,
                    no_ws_piece_offset=no_ws_offset,
                    ws_piece_offset=ws_offset,
                )
            )

            no_ws_offset += pieces_len
            ws_offset += pieces_len

        # There can be spaces after the last non-whitespace token.
        n_ws = doc_ws_counts[len(pieces_lens)]
        for i in range(ws_offset, ws_offset + n_ws):
            doc_alignments.append(
                TokenAlignment(ws_piece_offset=i, no_ws_piece_offset=None, n_pieces=1)
            )
        ws_offset += n_ws

        # Add the doc alignment.
        alignments.append(
            Alignment(
                tokens=doc_alignments,
                no_ws_n_pieces=no_ws_offset,
                ws_n_pieces=ws_offset,
            )
        )

    return alignments


def _filter_tokens(docs: List[Doc]) -> Tuple[Tok2PiecesInT, List[Counter]]:
    """Filter out whitespace tokens. Returns the non-whitespace tokens
    and a mapping from the (non-whitespace) token offset to the number
    of whitespaces that preceded the token."""
    # Deprecated
    tokens = []
    ws_counts = []
    for doc in docs:
        doc_tokens = []
        doc_ws_counts: Counter = Counter()
        offset = 0
        for token in doc:
            if token.is_space:
                doc_ws_counts[offset] += 1
                continue
            doc_tokens.append(token)
            offset += 1
        tokens.append(doc_tokens)
        ws_counts.append(doc_ws_counts)

    return tokens, ws_counts


def _add_whitespace_tokens(
    model: Model, Y_no_ws: WsTokenAdapterOutT, alignments: List[Alignment]
):
    """Add stub representations for whitespace tokens."""
    # Deprecated
    for Y_doc, doc_alignment in zip(Y_no_ws.all_outputs, alignments):
        if doc_alignment.has_no_whitespace:
            continue

        hidden_width = Y_doc[0].dataXd.shape[1]
        for layer_idx, layer in enumerate(Y_doc):
            lengths = []
            new_layer = model.ops.alloc2f(doc_alignment.ws_n_pieces, hidden_width)

            for alignment in doc_alignment:
                if not alignment.is_whitespace:
                    assert alignment.no_ws_piece_offset is not None
                    new_layer[
                        alignment.ws_piece_offset : alignment.ws_piece_offset
                        + alignment.n_pieces,
                        :,
                    ] = layer.dataXd[
                        alignment.no_ws_piece_offset : alignment.no_ws_piece_offset
                        + alignment.n_pieces,
                        :,
                    ]
                lengths.append(alignment.n_pieces)

            Y_doc[layer_idx] = Ragged(new_layer, lengths=model.ops.asarray1i(lengths))


def _remove_whitespace_tokens(
    model: Model, dY: List[List[Ragged]], alignments: List[Alignment]
):
    """Remove representations for whitespace tokens."""
    # Deprecated
    for dY_doc, doc_alignment in zip(dY, alignments):
        if doc_alignment.has_no_whitespace:
            continue

        hidden_width = cast(Tuple[int, ...], dY_doc[0].dataXd.shape)[1]
        for layer_idx, layer in enumerate(dY_doc):
            new_layer = model.ops.alloc2f(doc_alignment.no_ws_n_pieces, hidden_width)
            lengths = []

            for alignment in doc_alignment:
                if alignment.is_whitespace:
                    continue

                assert alignment.no_ws_piece_offset is not None
                # Extra type ignore to accomodate two MyPy versions :(.
                new_layer[  # type: ignore
                    alignment.no_ws_piece_offset : alignment.no_ws_piece_offset
                    + alignment.n_pieces,  # type: ignore
                    :,
                ] = layer.dataXd[  # type: ignore
                    alignment.ws_piece_offset : alignment.ws_piece_offset
                    + alignment.n_pieces,
                    :,
                ]

                lengths.append(alignment.n_pieces)

            dY_doc[layer_idx] = Ragged(new_layer, lengths=model.ops.asarray1i(lengths))
