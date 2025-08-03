// static/js/orb.js
import { Renderer, Program, Mesh, Triangle, Vec3 } from "https://cdn.jsdelivr.net/npm/ogl@0.0.32/dist/ogl.mjs";

document.addEventListener("DOMContentLoaded", () => {
  const container = document.getElementById("orb");
  if (!container) return;

  const vert = `
    precision highp float;
    attribute vec2 position;
    attribute vec2 uv;
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;

  const frag = `...`; // 🔥 Copy the full fragment shader from your React code here

  const renderer = new Renderer({ alpha: true, premultipliedAlpha: false });
  const gl = renderer.gl;
  gl.clearColor(0, 0, 0, 0);
  container.appendChild(gl.canvas);

  const geometry = new Triangle(gl);
  const program = new Program(gl, {
    vertex: vert,
    fragment: frag,
    uniforms: {
      iTime: { value: 0 },
      iResolution: {
        value: new Vec3(gl.canvas.width, gl.canvas.height, gl.canvas.width / gl.canvas.height),
      },
      hue: { value: 200 }, // 🎨 Customize hue
      hover: { value: 0 },
      rot: { value: 0 },
      hoverIntensity: { value: 0.3 },
    },
  });

  const mesh = new Mesh(gl, { geometry, program });

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const width = container.clientWidth;
    const height = container.clientHeight;
    renderer.setSize(width * dpr, height * dpr);
    gl.canvas.style.width = width + "px";
    gl.canvas.style.height = height + "px";
    program.uniforms.iResolution.value.set(gl.canvas.width, gl.canvas.height, gl.canvas.width / gl.canvas.height);
  }
  window.addEventListener("resize", resize);
  resize();

  let lastTime = 0;
  let currentRot = 0;
  const rotationSpeed = 0.3;

  function animate(t) {
    requestAnimationFrame(animate);
    const dt = (t - lastTime) * 0.001;
    lastTime = t;

    program.uniforms.iTime.value = t * 0.001;
    currentRot += dt * rotationSpeed;
    program.uniforms.rot.value = currentRot;

    renderer.render({ scene: mesh });
  }
  requestAnimationFrame(animate);
});
