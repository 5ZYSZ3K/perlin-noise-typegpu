# Overview

This repository is a perlin-noise generator written in [TypeGPU](https://github.com/software-mansion/TypeGPU).

# Implementation

## Technologies

It uses Typescript, Vite, WebGPU (TypeGPU layer above it), and shaders written in WGSL (embeded in Typescript code)

## Description

It uses 3 shaders: for computation, for rendering and for shading
First of all, a value map is being generated, which is then passed to rendering sharer, responsible for mapping each value to its pixel. At the end, everything lands in shading shader responsible for picking colors.

## Running locally

You will need a package manager for that (this repo uses `yarn`). Clone the repo, install packages, and run it using command `yarn start` (or `<your-package-manager-run-command> start`)
