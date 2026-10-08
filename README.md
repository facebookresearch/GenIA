<!-- Copyright (c) Meta Platforms, Inc. and affiliates. -->

# GenIA Project Page

This branch contains the website for **GenIA: Generative Reconstruction with
Test-Time Input Alignment**, including qualitative comparisons, benchmark tables,
and the 3D viewer integration. The reconstruction code is on the `release` branch.

## Local Preview

From this branch's checkout, serve the site with Python 3:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open <http://127.0.0.1:8000>. The pages use the committed JavaScript data;
no build step is required. On localhost, media is loaded from the local assets
folder, which is not included in Git. The published site uses hosted media,
and the 3D viewer loads an external bundle.

The data-generation scripts depend on the authors' research tooling and source
results, which are not included in this branch. They are not required to serve
the existing pages. Do not run the upload script just to preview the site.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

Project-written website code and documentation are licensed under **CC BY-NC 4.0**;
see [LICENSE](LICENSE). Third-party components, embedded fonts, and referenced
media retain their own licenses and notices.
