import logo from './logo.svg';
import { Navigation, Footer, Home, About} from "./pages/index";
import './App.css';
import React, { Component } from 'react';
import { BrowserRouter as Router, Route, Switch, Link, Redirect } from 'react-router-dom';

function App() {
  return (
    <div className="App">
      <Router>
        <Navigation/>
        <Switch>
          <Route path="/" exact component={() => <Home/>} />
          <Route path="/about" exact component={() => <About/>} />
          <initClient></initClient>
        </Switch>
        <Footer/>
      </Router>
    </div>
  );
}



export default App;
