"""DeepLegis Models"""
import typing
import tensorflow as tf
#from tensorflow.keras.layers import Input, Dense, concatenate, Dropout

from transformers.models.longformer.modeling_tf_longformer import * #TFLongformerModelPreTrainedModel, LongformerConfig
        
class DeepLegisClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size+1,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.out_proj = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )

    def call(self, hidden_states, partisan_lean, training=False):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = tf.keras.layers.concatenate([hidden_states, partisan_lean])
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        output = self.out_proj(hidden_states)
        return output


class DeepLegis(TFLongformerPreTrainedModel, DeepLegisClassificationHead):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        self.longformer = TFLongformerMainLayer(config, add_pooling_layer=False, name="longformer")
        self.classifier = DeepLegisClassificationHead(config, name="classifier")

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        partisan_lean=None,
        token_type_ids=None,
        position_ids=None,
        global_attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["global_attention_mask"] is None and inputs["input_ids"] is not None:
            logger.info("Initializing global attention on CLS token...")
            # global attention on cls token
            inputs["global_attention_mask"] = tf.zeros_like(inputs["input_ids"])
            updates = tf.ones(shape_list(inputs["input_ids"])[0], dtype=tf.int32)
            indices = tf.pad(
                tensor=tf.expand_dims(tf.range(shape_list(inputs["input_ids"])[0]), axis=1),
                paddings=[[0, 0], [0, 1]],
                constant_values=0,
            )
            inputs["global_attention_mask"] = tf.tensor_scatter_nd_update(
                inputs["global_attention_mask"],
                indices,
                updates,
            )

        outputs = self.longformer(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            global_attention_mask=inputs["global_attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        sequence_output = outputs[0]

        logits = self.classifier(sequence_output, partisan_lean, training=inputs['training'])

        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], logits)

        if not inputs["return_dict"]:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFLongformerSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )


    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None
        g_attns = tf.convert_to_tensor(output.global_attentions) if self.config.output_attentions else None

        return TFLongformerSequenceClassifierOutput(
            logits=output.logits, hidden_states=hs, attentions=attns, global_attentions=g_attns
        )
