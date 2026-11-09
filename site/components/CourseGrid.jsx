import Link from 'next/link';
import { Card, Tag } from './ui';

export const CourseGrid = ({ items, tag }) => (
  <div className="grid">
    {items.map(l => (
      <Card as={Link} key={l.id} href={`/lectures/${l.id}`}>
        {tag && <Tag>{tag}</Tag>}
        <h3>{l.title}</h3>
        <p>{l.explorers.length} interactive sections</p>
      </Card>
    ))}
  </div>
);
