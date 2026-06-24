/**
 * The model's class labels, in output-index order.
 *
 * Mirrors the 26-class A–Z label order recorded in the trained checkpoint
 * (`class_names` in `artifacts/checkpoints/best_model.pth`) and reproduced by
 * the ONNX export. Index i of the model's logits corresponds to CLASS_NAMES[i].
 *
 * This dataset (Marxulia/asl_sign_languages_alphabets_v03) has no del / nothing
 * / space control signs — only the 26 letters.
 */
export const CLASS_NAMES: readonly string[] = [
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H",
  "I",
  "J",
  "K",
  "L",
  "M",
  "N",
  "O",
  "P",
  "Q",
  "R",
  "S",
  "T",
  "U",
  "V",
  "W",
  "X",
  "Y",
  "Z",
];

export const NUM_CLASSES = CLASS_NAMES.length;
