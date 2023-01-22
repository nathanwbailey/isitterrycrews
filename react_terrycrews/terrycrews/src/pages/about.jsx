import react from "react"

function About() {
    return (
        <div className="App">
            <header className="App-body">
                <h1>Is it Terry Crews?</h1>
                <br></br>
                <br></br>
                <p>A joke project I made back in 2022 that I thought needed to be published</p>
                <br></br>
                <p> Uses a mobilenetv3 network into a simple binary classifier to output if the image contains Terry or not </p>
                <br></br>
                <p> All code is avaliable here including training graphs and outcomes </p>
            </header>
        </div>
    );
}

export default About;