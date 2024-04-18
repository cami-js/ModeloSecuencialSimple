document.addEventListener('DOMContentLoaded', function() {
    let model;
    let entrenamientoCompletado = false;

    // Crear el modelo secuencial
    function crearModelo() {
        model = tf.sequential();
        model.add(tf.layers.dense({units: 1, inputShape: [1]}));
        model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
    }

    // Entrenamiento del modelo
    async function entrenarModelo() {
        const botonEntrenar = document.getElementById('entrenarBtn');
        botonEntrenar.disabled = true; 
        const xs = tf.tensor2d([[-6], [-5], [-4], [-3], [-2], [-1], [0], [1], [2]]);
        const ys = tf.tensor2d([[-6], [-4], [-2], [0], [2], [4], [6], [8], [10]]); // Valores de y correspondientes a la función y = 2x + 6
        const history = await model.fit(xs, ys, {epochs: 1000, callbacks: tfvis.show.fitCallbacks(
            {name: 'Entrenamiento'}, 
            ['loss']            
        )});
        entrenamientoCompletado = true;
        const mensajeElement = document.getElementById('mensajeEntrenamiento');
        mensajeElement.innerText = 'Entrenamiento completado.';
    }

    // Función para predecir Y
    async function predecirY() {
        if (!entrenamientoCompletado) {
            alert('Por favor, entrena el modelo antes de realizar predicciones.');
            return;
        }
        const xInput = parseFloat(document.getElementById('xInput').value);
        const xNew = tf.tensor2d([[xInput]]);
        const yPred = model.predict(xNew);
        const resultadoElement = document.getElementById('resultado');
        resultadoElement.innerText = `El valor de Y predicho para X = ${xInput} es: ${yPred.dataSync()[0]}`;
    }

    // Event listener para iniciar el entrenamiento cuando se hace clic en el botón "Entrenar modelo"
    document.getElementById('entrenarBtn').addEventListener('click', function() {
        crearModelo();
        entrenarModelo();
    });

    // Event listener para predecir Y cuando se hace clic en el botón "Predecir"
    document.getElementById('predecirBtn').addEventListener('click', predecirY);
});
