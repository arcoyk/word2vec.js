
https://user-images.githubusercontent.com/5988574/121248622-9c916000-c8de-11eb-96a0-6a9cfee806f0.mov



```
clone git@github.com:arcoyk/word2vec.js.git
// open index.html for demo
```

# word2vec.js
Light weight frontend Word2Vec with sample trained model of 3171 Japanese words. 3171 words were randomly selected from original Word2Vec model (330,000 words, 50 dimentional vectors, trained with Wikipedia articles).

- Word count: 3171
- Vector dimention: 50
- Language: Japanese
- Original source: Wikipedia

Pretrained model was translated into words and vectors which can be found at ./js/data50.js. This will be ready as soon as you open index.html. Edit this file to change the model.

# Example
```js
// most_similar equivalent
word2vec.w2ws("両極性").ws
// => ["両極端", "メタ理論", "人生観", "欲張り", "主体性"] 
```

Note that the output is an Object that consists similar words and similarities.
```js
let result = word2vec.w2ws("補修")
// result = {
//     ws: ["補修", "点検", "堰堤", "バラストタンク", "人手"],
//     ps: [1.0000001192092896, 0.7502378225326538, 0.6257367134094238, 0.6090835928916931, 0.5916973352432251]
// }
// similarity between "補修" and "点検" is 0.75023
```

Offcourse, you can get the vector
```js
word2vec.w2v("補修")
// [-0.14429277181625366, 0.18658098578453064, 0.1847711056470871, 0.09209378808736801, -0.0679391548037529, 0.16327759623527527, -0.031874626874923706, -0.16484060883522034, -0.06378225237131119, 0.023497363552451134, 0.0020464088302105665, 0.17774838209152222, 0.02699054591357708, 0.16259080171585083, 0.28372877836227417, 0.07724079489707947, -0.2403295338153839, -0.0819578692317009, 0.20890462398529053, -0.028673263266682625, 0.009770154021680355, 0.13413363695144653, 0.011923604644834995, -0.11731985211372375, -0.10551074147224426, 0.08331461250782013, 0.004097146913409233, 0.24513141810894012, -0.030268225818872452, -0.01968240551650524, 0.01812598668038845, 0.06748735904693604, 0.09437521547079086, -0.0255605336278677, 0.01223302073776722, -0.016752682626247406, -0.0056835669092834, 0.008190684020519257, -0.4212040603160858, -0.0371280238032341, 0.1373198926448822, -0.1416447013616562, -0.12562932074069977, 0.3315233290195465, 0.12790662050247192, -0.11261098831892014, -0.011190776713192463, -0.17273341119289398, 0.15604254603385925, 0.17403298616409302]
```

# Why
[ml5.js](https://ml5js.org/reference/api-Word2vec/) provides more stable word2vec on js with full features. word2vec.js was made for ready-to-use repo with a hostable-size model for my personal convenience. 

word2vec.js differs...

1. word2vec.js accepts separated model: easy to create tensor matrix with TensorFlow.js, easy to set attribute on WebGL shader (in case you want to visualize it).

```js
// common word2vec model
{"dog": [0.32, 0.43, 0.45], "cat": [0.12, 0.23, 0.54]}
// word2vec.js accepts separated model
{"ws": ["dog", "cat"], "vs": [[0.32, 0.43, 0.45], [0.12, 0.23, 0.54]]}
```

2. Calculate cosine similarity by matrix dot (which is clean and slightly fast (maybe)). 

```js
// Calculation of cosine similarity (for all vectors) can be boiled down to a simple matrix dot which will be done by TensorFlow
word2vec.w2ws = function(w, topn=5) {
    let v = word2vec.w2v(w)
    v = tf.tensor(v)
    v = v.reshape([v.shape[0], 1])
    let p = tf.matMul(Data.tensor.vs, v)
    p = p.flatten()
    let ids = tf.topk(p, topn).indices
    let ws = ids.arraySync().map(i => Data.ws[i])
    let ps = ids.arraySync().map(i => p.arraySync()[i])
    return {ws:ws, ps:ps}
}
```
