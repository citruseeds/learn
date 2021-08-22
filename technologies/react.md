# React
## Return to
* 16 (What are synthetic events in React?)

## Components
function definition;
```jsx
function Greeting({ message }) {
  return <h1>{`Hello, ${message}`}</h1>
}
```

class definition;
```jsx
class Greeting extends React.Component {
  render() {
    return <h1>{`Hello, ${this.props.message}`}</h1>
  }
}
```

prefer class definition if lifecycle methods (which are?) or state is needed, else prefer function definition; although with hooks you can access them in function def,, so always pref function def?

pure component vs reg component; `exactly the same as React.Component except that it handles the shouldComponentUpdate() method for you`

## State
use `setState()` to trigger rerender (this.state sets wont trigger); setstate has a callback, but should use lifecycle method (which?) instead

## Events
binding