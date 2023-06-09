// import io from "socket.io-client";

// export const socket = io();

// export const socket = io('http://localhost:5000/')
var socket = io.connect();
var total = new Number();
var mask = new Number();
var no_mask = new Number();
var incorrect = new Number();
var isCrownded = new Boolean();

socket.on("new_user", function () {
  socket.emit("Connecting new user", { new_user: { username: "Vi" } });
});

socket.on("Sending result", function (msg) {
  mask = msg.mask;
  no_mask = msg.no_mask;
  incorrect = msg.incorrect;
  total = msg.total;
  document.getElementById("total").innerHTML = total;
  document.getElementById("mask").innerHTML = mask;
  document.getElementById("no_mask").innerHTML = no_mask;
  document.getElementById("incorrect").innerHTML = incorrect;
  document.getElementById("inference_time").innerHTML = msg.inference_time;
  // if (mask < no_mask + incorrect) {
  //   isCrownded = true;
  // } else isCrownded = false;
  $(document).ready(function () {
    $("#warning").toggle(
      mask < no_mask + incorrect && mask + no_mask + incorrect > 1
    );
  });
});
