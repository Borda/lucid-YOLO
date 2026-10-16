# 📦 WP-148 — `test(data): derived and invariant expectations for every transform`

> Engineering log · [Phase 13 — Contributor admission, augmentation guard and keypoint parity, v0.6.0](https://borda.github.io/lucid-YOLO/engineering/phase-13-contributor-admission/)
>
> 🔗 Depends on [WP-147](https://borda.github.io/lucid-YOLO/work_packages/wp-147/) · Needed by [WP-149](https://borda.github.io/lucid-YOLO/work_packages/wp-149/)
>
> ⬅️ [WP-147](https://borda.github.io/lucid-YOLO/work_packages/wp-147/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-149](https://borda.github.io/lucid-YOLO/work_packages/wp-149/) ➡️


## WP-148 — the tier that proves rather than pins

<a id="wp-148"></a>

`tests/data/test_aug_contract.py` holds twenty-two cases across two tiers and not one characterization value, which is the property that makes it reviewable: every expectation in it is derivable on paper from the transform's own definition, so a reader checks the arithmetic rather than trusting that a previous run printed this.

**Tier A is the only tier that catches a bug rather than a change.** A frozen output pins whatever the code does today, including whatever it does wrong; a derived expectation says what the code *should* do. Every case that geometry allows was written into this tier rather than left for tier B: an identity affine returns its input, an integer-pixel translation is an exact shifted copy, a quarter turn sends the box `[1,2,3,6]` to `[2,1,6,3]`, a scale of two about the centre sends `[2,2,4,4]` to `[0,0,4,4]`, a 10×20 source letterboxes at ratio 0.8 with four rows of padding, forward-then-inverse returns the points exactly, two mirrors compose to the identity including the keypoint permutation, a stated mosaic centre puts one box at four derived offsets, mixup at `lam=0.25` on constants is exactly 0.25.

**The quarter-turn case is worth the arithmetic it took.** On an eight-pixel canvas the centre is `(4, 4)`, and the rotation maps `(x, y)` to `(4 - (y - 4), 4 + (x - 4))`, so the corners `(1, 2)` and `(3, 6)` land on `(6, 1)` and `(2, 3)` and the envelope is `[2, 1, 6, 3]`. The image under the same rotation is not `torch.rot90` of the input — it is `rot90` in the other direction, rolled by one column, because an even canvas has its centre on a pixel *corner* rather than a pixel centre. That asymmetry is real and correct and would have been easy to write down wrongly as an expectation; the box, whose mapping is a matrix and not a resampling, is the honest thing to assert. Pixels stay in the cases where they are analytically forced.

**Tier C constrains the modalities against each other, which is where a shape-compatible wrong answer hides.** A translation that pushes one of two instances off the canvas must drop it from boxes, labels, polygons, keypoints *and* visibilities together; dropping it from boxes alone leaves a keypoint set supervising the surviving box with the departed instance's points, and every shape check still passes. The alignment invariant — a box re-derived from a transported ring equals the box transported separately — is the property the whole ring route will rest on once the warp moves upstream, and it holds only while both ride one matrix. Rotated boxes leave every transform in canonical long-edge form, because that is a task convention rather than a resampling property and has to be re-established after each warp. Out-of-canvas pixels are exactly the grey fill, not zero. Mixup concatenates both label sets unweighted, since the blend factor weighs pixels and never labels.

**The keypoint invariant is the one that ties two transports together.** A point marking a bright pixel must still mark it after the warp: an integer translation moves the pixel and the point by the same three-and-two, and the assertion reads the warped image at the warped point. A keypoint path using a different sign or a different centre convention from the image path would move the point off its feature while both outputs stayed well inside the canvas and every shape matched.

One case needed its parameters adjusted rather than its expectation: the first draft translated by fourteen pixels to strand one of two instances, and stranded both — the surviving box clipped to a one-pixel width and fell under the default `min_box_size` of two. Ten pixels leaves one instance intact. Worth recording because the invariant itself passed either way: with both instances dropped, every modality still agreed on zero, and the test was green while checking nothing. An invariant over an empty set is a case that cannot fail, which is the most expensive kind of passing test.

Coverage reaches every transform, including the ones staying local — `HSVJitter`, the assemblies, and A11's half-open rasterisation, whose ring spanning `[1,4]` covers three pixels rather than four. A transform that stays is still refactored around, and a guard written after the refactor guards nothing.

